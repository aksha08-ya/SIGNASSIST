"""
Dynamic temporal sequence buffer and tensor builder for SIGNASSIST.

Maintains sliding temporal windows and constructs canonical dynamic sequence tensors
with shape [B, 5, T, 115], supporting sequence padding, mask generation, and no NaNs.
"""

from collections import deque
from typing import List, Optional
import numpy as np

from src.features.constants import (
    DEFAULT_SEQUENCE_LENGTH,
    NUM_FEATURE_CHANNELS,
    NUM_GRAPH_NODES,
)
from src.features.types import DynamicSequenceTensor, SelectedGraphLandmarks


class SequenceBuffer:
    """Sliding temporal sequence window buffer producing [B, 5, T, 115] tensors."""

    def __init__(self, sequence_length: int = DEFAULT_SEQUENCE_LENGTH) -> None:
        self.sequence_length = sequence_length
        self._buffer: deque = deque(maxlen=self.sequence_length)
        self._mask_buffer: deque = deque(maxlen=self.sequence_length)

    def reset(self) -> None:
        """Clears the temporal frame buffer."""
        self._buffer.clear()
        self._mask_buffer.clear()

    def add_frame(self, frame_landmarks: SelectedGraphLandmarks) -> None:
        """Appends a normalized frame (115, 5) and its mask (115,) to the sliding window."""
        self._buffer.append(frame_landmarks.nodes.copy())
        self._mask_buffer.append(frame_landmarks.mask.copy())

    def __len__(self) -> int:
        return len(self._buffer)

    def get_tensor(self, batch_size: int = 1) -> DynamicSequenceTensor:
        """Constructs the canonical dynamic sequence tensor [B, 5, T, 115].

        Padding rule for incomplete sequences (< T frames):
        Left-pads with zeroed frame nodes and marks corresponding temporal mask entries as False.

        Args:
            batch_size: Batch dimension B (default 1).

        Returns:
            DynamicSequenceTensor containing:
            - tensor: np.ndarray shape (B, 5, T, 115) float32
            - mask: np.ndarray shape (B, T, 115) bool
            - valid_frames_count: int
        """
        curr_len = len(self._buffer)
        T = self.sequence_length
        V = NUM_GRAPH_NODES
        C = NUM_FEATURE_CHANNELS

        # Prepare temporal frame sequence of shape (T, V, C)
        sequence_data = np.zeros((T, V, C), dtype=np.float32)
        sequence_mask = np.zeros((T, V), dtype=bool)

        if curr_len > 0:
            # Place buffered frames at the end (left-padding)
            start_idx = T - curr_len
            frames_arr = np.array(self._buffer, dtype=np.float32)  # (curr_len, V, C)
            masks_arr = np.array(self._mask_buffer, dtype=bool)  # (curr_len, V)

            sequence_data[start_idx:] = frames_arr
            sequence_mask[start_idx:] = masks_arr

        # Permute from (T, V, C) to channels-first (C, T, V) = (5, T, 115)
        # Transpose mapping: 0(T), 1(V), 2(C) -> 2(C), 0(T), 1(V)
        single_tensor = np.transpose(sequence_data, (2, 0, 1))  # (5, T, 115)

        # Guarantee no NaN / Inf values
        single_tensor = np.nan_to_num(single_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # Expand batch dimension B: (B, 5, T, 115)
        batch_tensor = np.repeat(single_tensor[np.newaxis, ...], batch_size, axis=0)
        batch_mask = np.repeat(sequence_mask[np.newaxis, ...], batch_size, axis=0)

        return DynamicSequenceTensor(
            tensor=batch_tensor,
            mask=batch_mask,
            valid_frames_count=curr_len,
            sequence_length=T,
        )
