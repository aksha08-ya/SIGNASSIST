"""Continuous Sign Boundary Detection & Segmentation for SIGNASSIST (Phase 9).

This module implements continuous sign boundary detection using motion-energy minima,
rest-pose classification, temporal smoothing, and hysteresis debouncing.

Architecture:
- Inputs dynamic landmark tensor [B, 5, T, 115]
- Reuses pre-computed vx/vy channels (channels 3 and 4)
- Computes per-frame aggregate active node velocity magnitude (motion energy)
- Non-recurrent temporal smoothing & hysteresis filtering via TemporalSmoother
- Identifies rest poses, energy minima, candidate boundaries, and sign segments
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pipeline.temporal_smoother import TemporalSmoother
from src.utils.config import BoundaryConfig, ModelConfig


@dataclass
class SignSegment:
    """Represents a continuous sign segment within a sequence.

    Attributes:
        start_frame: Index of starting frame (0-indexed).
        end_frame: Index of ending frame (inclusive).
        duration: Total number of frames in segment (end_frame - start_frame + 1).
        mean_energy: Average smoothed motion energy during the segment.
        boundary_evidence: Bounded 0..1 scalar measuring segmentation/boundary motion
            evidence relative to energy threshold. NOTE: This is motion/segmentation evidence
            and must NOT be confused with Phase 8 classifier recognition confidence.
        is_valid: Flag indicating if segment satisfies duration and quality criteria.
    """

    start_frame: int
    end_frame: int
    duration: int
    mean_energy: float
    boundary_evidence: float
    is_valid: bool = True


@dataclass
class BoundaryDetectionOutput:
    """Structured output from SignBoundaryDetector."""

    segments: List[List[SignSegment]]
    motion_energy: torch.Tensor  # [B, T] raw motion energy
    smoothed_energy: torch.Tensor  # [B, T] smoothed motion energy
    boundary_flags: torch.Tensor  # [B, T] boolean or binary tensor of boundary triggers
    rest_pose_mask: torch.Tensor  # [B, T] boolean tensor of rest-pose frames


class SignBoundaryDetector(nn.Module):
    """Detects continuous sign boundaries and extracts sign segments from landmark sequences.

    Uses kinematic motion energy (velocity magnitude across active nodes) combined with
    temporal smoothing, rest-pose thresholds, local minima detection, and hysteresis debouncing.
    """

    def __init__(
        self,
        config: Optional[Union[BoundaryConfig, ModelConfig]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(config, ModelConfig):
            b_cfg = config.boundary_detection
        elif isinstance(config, BoundaryConfig):
            b_cfg = config
        elif config is None:
            b_cfg = BoundaryConfig()
        else:
            raise TypeError(
                f"Expected config to be BoundaryConfig or ModelConfig, got {type(config)}"
            )

        # Allow kwargs override
        self.energy_smoothing_window = kwargs.get(
            "energy_smoothing_window", b_cfg.energy_smoothing_window
        )
        self.energy_threshold = kwargs.get("energy_threshold", b_cfg.energy_threshold)
        self.rest_pose_threshold = kwargs.get(
            "rest_pose_threshold", b_cfg.rest_pose_threshold
        )
        self.min_segment_duration = kwargs.get(
            "min_segment_duration", b_cfg.min_segment_duration
        )
        self.min_boundary_separation = kwargs.get(
            "min_boundary_separation", b_cfg.min_boundary_separation
        )
        self.debounce_window = kwargs.get("debounce_window", b_cfg.debounce_window)
        self.active_nodes = kwargs.get("active_nodes", b_cfg.active_nodes)

        self.smoother = TemporalSmoother()

    def compute_motion_energy(
        self,
        landmarks: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes aggregate scalar motion energy per frame from active landmark nodes.

        Args:
            landmarks: Dynamic landmark tensor of shape [B, 5, T, 115].
            valid_mask: Optional boolean tensor of shape [B, T, 115] indicating valid nodes.

        Returns:
            Scalar motion energy tensor of shape [B, T].
        """
        if landmarks.dim() != 4 or landmarks.shape[1] != 5 or landmarks.shape[3] != 115:
            raise ValueError(
                f"Expected landmarks tensor of shape [B, 5, T, 115], got {list(landmarks.shape)}"
            )

        B, C, T, V = landmarks.shape

        # Extract vx (channel 3) and vy (channel 4) for active nodes
        active_indices = [idx for idx in self.active_nodes if 0 <= idx < V]
        if not active_indices:
            return torch.zeros((B, T), device=landmarks.device, dtype=landmarks.dtype)

        vx = landmarks[:, 3, :, active_indices]  # [B, T, num_active]
        vy = landmarks[:, 4, :, active_indices]  # [B, T, num_active]

        # Velocity magnitude per active node
        speed = torch.sqrt(vx**2 + vy**2 + 1e-8)  # [B, T, num_active]

        # Determine node validity
        if valid_mask is not None:
            active_valid = valid_mask[:, :, active_indices].to(speed.dtype)  # [B, T, num_active]
        else:
            # Check if landmark coordinates are non-zero
            x = landmarks[:, 0, :, active_indices]
            y = landmarks[:, 1, :, active_indices]
            active_valid = ((torch.abs(x) > 1e-6) | (torch.abs(y) > 1e-6)).to(speed.dtype)

        # Mask invalid node speeds
        valid_speed = speed * active_valid  # [B, T, num_active]

        # Mean speed across valid active nodes (zero-division safe)
        valid_count = active_valid.sum(dim=-1)  # [B, T]
        sum_speed = valid_speed.sum(dim=-1)  # [B, T]

        energy = sum_speed / torch.clamp(valid_count, min=1.0)
        # If no valid active nodes, energy is 0
        energy = torch.where(valid_count > 0, energy, torch.zeros_like(energy))

        return energy

    def detect_boundaries(
        self,
        landmarks: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> BoundaryDetectionOutput:
        """Detects sign boundaries and segments input landmark sequence.

        Args:
            landmarks: Dynamic landmark tensor [B, 5, T, 115].
            valid_mask: Optional validity mask [B, T, 115].

        Returns:
            BoundaryDetectionOutput structured object.
        """
        raw_energy = self.compute_motion_energy(landmarks, valid_mask=valid_mask)  # [B, T]
        B, T = raw_energy.shape

        if T == 0:
            return BoundaryDetectionOutput(
                segments=[[] for _ in range(B)],
                motion_energy=raw_energy,
                smoothed_energy=raw_energy,
                boundary_flags=torch.zeros((B, 0), dtype=torch.bool, device=landmarks.device),
                rest_pose_mask=torch.zeros((B, 0), dtype=torch.bool, device=landmarks.device),
            )

        # 1. Temporal smoothing of motion energy
        smoothed_energy = self.smoother.smooth_motion_energy(
            raw_energy, window_size=self.energy_smoothing_window
        )  # [B, T]

        # 2. Rest-pose classification
        rest_pose_mask = smoothed_energy < self.rest_pose_threshold  # [B, T]

        # 3. Local energy minima candidate detection
        candidate_boundaries = torch.zeros((B, T), dtype=torch.bool, device=landmarks.device)
        for b in range(B):
            se = smoothed_energy[b]
            for t in range(1, T - 1):
                # Local minimum condition below energy threshold
                if (
                    se[t] <= se[t - 1]
                    and se[t] <= se[t + 1]
                    and se[t] < self.energy_threshold
                ):
                    candidate_boundaries[b, t] = True
            # Also treat rest-pose transitions as candidate boundaries
            candidate_boundaries[b] = candidate_boundaries[b] | rest_pose_mask[b]

        # 4. Hysteresis debouncing of candidate boundaries
        boundary_flags = self.smoother.debounce_events(
            candidate_boundaries, min_separation=self.min_boundary_separation
        ).to(torch.bool)  # [B, T]

        # 5. Extract sign segments
        all_segments: List[List[SignSegment]] = []
        for b in range(B):
            sample_segments: List[SignSegment] = []
            se = smoothed_energy[b]

            # Identify contiguous non-rest-pose active regions
            active_frames = (se >= self.rest_pose_threshold).cpu().numpy()

            in_segment = False
            start_f = 0

            for t in range(T):
                is_active = active_frames[t]
                is_boundary = bool(boundary_flags[b, t])

                if not in_segment:
                    if is_active and not is_boundary:
                        in_segment = True
                        start_f = t
                else:
                    # End segment if hit rest pose or debounced boundary trigger
                    if not is_active or is_boundary:
                        end_f = t - 1 if not is_active else t
                        if end_f >= start_f:
                            duration = end_f - start_f + 1
                            if duration >= self.min_segment_duration:
                                seg_energy = float(se[start_f : end_f + 1].mean().item())
                                boundary_ev = min(1.0, seg_energy / (self.energy_threshold + 1e-6))
                                sample_segments.append(
                                    SignSegment(
                                        start_frame=start_f,
                                        end_frame=end_f,
                                        duration=duration,
                                        mean_energy=seg_energy,
                                        boundary_evidence=boundary_ev,
                                        is_valid=True,
                                    )
                                )
                        in_segment = False

            # Check trailing segment at end of sequence
            if in_segment:
                end_f = T - 1
                duration = end_f - start_f + 1
                if duration >= self.min_segment_duration:
                    seg_energy = float(se[start_f : end_f + 1].mean().item())
                    boundary_ev = min(1.0, seg_energy / (self.energy_threshold + 1e-6))
                    sample_segments.append(
                        SignSegment(
                            start_frame=start_f,
                            end_frame=end_f,
                            duration=duration,
                            mean_energy=seg_energy,
                            boundary_evidence=boundary_ev,
                            is_valid=True,
                        )
                    )

            all_segments.append(sample_segments)

        return BoundaryDetectionOutput(
            segments=all_segments,
            motion_energy=raw_energy,
            smoothed_energy=smoothed_energy,
            boundary_flags=boundary_flags,
            rest_pose_mask=rest_pose_mask,
        )

    def forward(
        self,
        landmarks: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> BoundaryDetectionOutput:
        """Callable forward wrapper for boundary detection."""
        return self.detect_boundaries(landmarks, valid_mask=valid_mask)
