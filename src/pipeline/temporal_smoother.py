"""Temporal smoothing and hysteresis filtering for continuous sign boundary detection.

This module provides deterministic, non-recurrent temporal filtering utilities:
- Motion energy smoothing via moving average (odd window size required)
- Class/event prediction majority voting
- Boundary event debouncing / hysteresis filtering
"""

from typing import Union
import torch
import torch.nn.functional as F


class TemporalSmoother:
    """Non-recurrent temporal smoother for sign boundary signals."""

    @staticmethod
    def smooth_motion_energy(
        energy: torch.Tensor, window_size: int = 5
    ) -> torch.Tensor:
        """Smooths a 1D or 2D motion energy tensor along the temporal dimension.

        Args:
            energy: Tensor of shape [B, T] or [T].
            window_size: Size of moving average window (must be a positive odd integer).

        Returns:
            Smoothed motion energy tensor of identical shape [B, T] or [T].

        Raises:
            ValueError: If energy shape is invalid or window_size is not a positive odd integer.
        """
        if energy.dim() not in (1, 2):
            raise ValueError(
                f"Expected energy tensor of dim 1 [T] or 2 [B, T], got shape {list(energy.shape)}"
            )

        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be a positive odd integer, got {window_size}"
            )

        if window_size == 1:
            return energy.clone()

        is_1d = energy.dim() == 1
        if is_1d:
            energy_2d = energy.unsqueeze(0)  # [1, T]
        else:
            energy_2d = energy

        B, T = energy_2d.shape
        if T == 0:
            return energy.clone()

        pad = window_size // 2
        # Use reflection padding if T > pad, else replicate padding
        if T > pad:
            padded = F.pad(energy_2d.unsqueeze(1), (pad, pad), mode="reflect")  # [B, 1, T + 2*pad]
        else:
            padded = F.pad(energy_2d.unsqueeze(1), (pad, pad), mode="replicate")

        # 1D Average Pooling
        smoothed = F.avg_pool1d(padded, kernel_size=window_size, stride=1)  # [B, 1, T]
        smoothed = smoothed.squeeze(1)  # [B, T]

        if is_1d:
            return smoothed.squeeze(0)
        return smoothed

    @staticmethod
    def majority_vote(
        predictions: torch.Tensor, window_size: int = 5
    ) -> torch.Tensor:
        """Applies temporal majority voting (mode filter) across window_size frames.

        Args:
            predictions: Tensor of shape [B, T] or [T] containing discrete indices or flags.
            window_size: Odd integer window size for majority vote.

        Returns:
            Voted predictions tensor of identical shape and dtype.

        Raises:
            ValueError: If predictions shape is invalid or window_size is not a positive odd integer.
        """
        if predictions.dim() not in (1, 2):
            raise ValueError(
                f"Expected predictions tensor of dim 1 [T] or 2 [B, T], got shape {list(predictions.shape)}"
            )

        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be a positive odd integer, got {window_size}"
            )

        if window_size == 1:
            return predictions.clone()

        is_1d = predictions.dim() == 1
        if is_1d:
            preds_2d = predictions.unsqueeze(0)  # [1, T]
        else:
            preds_2d = predictions

        B, T = preds_2d.shape
        half_w = window_size // 2

        voted = preds_2d.clone()
        for b in range(B):
            for t in range(T):
                t_start = max(0, t - half_w)
                t_end = min(T, t + half_w + 1)
                window_vals = preds_2d[b, t_start:t_end]
                # torch.mode returns (values, indices)
                mode_val = torch.mode(window_vals).values
                voted[b, t] = mode_val

        if is_1d:
            return voted.squeeze(0)
        return voted

    @staticmethod
    def debounce_events(
        events: torch.Tensor, min_separation: int = 8
    ) -> torch.Tensor:
        """Enforces minimum temporal separation between triggered events (hysteresis debounce).

        Args:
            events: Boolean or integer binary tensor [B, T] or [T].
            min_separation: Minimum frame separation required between consecutive events.

        Returns:
            Debounced tensor of identical shape and dtype.
        """
        if events.dim() not in (1, 2):
            raise ValueError(
                f"Expected events tensor of dim 1 [T] or 2 [B, T], got shape {list(events.shape)}"
            )

        if min_separation <= 1:
            return events.clone()

        is_1d = events.dim() == 1
        if is_1d:
            events_2d = events.unsqueeze(0)  # [1, T]
        else:
            events_2d = events

        B, T = events_2d.shape
        debounced = torch.zeros_like(events_2d)

        for b in range(B):
            last_event_t = -min_separation
            for t in range(T):
                if bool(events_2d[b, t]):
                    if t - last_event_t >= min_separation:
                        debounced[b, t] = events_2d[b, t]
                        last_event_t = t

        if is_1d:
            return debounced.squeeze(0)
        return debounced
