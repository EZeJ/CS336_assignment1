from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch


class ActivationLogger:
    """
    Lightweight activation sampler for GP datasets.

    Stores flattened samples per (epoch, signal) key with an optional cap
    on the number of samples retained from each logging call.
    """

    def __init__(self, max_samples_per_call: int = 10_000, rng_seed: int = 0) -> None:
        self.max_samples_per_call = max_samples_per_call
        self.rng = np.random.default_rng(rng_seed)
        self._buffers: Dict[Tuple[int | None, str], list[np.ndarray]] = defaultdict(list)

    def record(self, signal: str, tensor: torch.Tensor, epoch: int | None = None) -> None:
        """
        Record a flattened view of `tensor`, subsampling if needed.

        Args:
            signal: name of the signal (e.g., 'silu_in', 'gate_u').
            tensor: torch.Tensor to log.
            epoch: optional epoch index to partition logs.
        """
        with torch.no_grad():
            arr = tensor.detach().cpu().reshape(-1).numpy()
        if arr.size > self.max_samples_per_call:
            idx = self.rng.choice(arr.size, size=self.max_samples_per_call, replace=False)
            arr = arr[idx]
        self._buffers[(epoch, signal)].append(arr)

    def get_epoch_buffers(self, epoch: int | None) -> dict[str, np.ndarray]:
        """Return concatenated arrays for all signals logged under the given epoch."""
        out = {}
        for (ep, sig), chunks in self._buffers.items():
            if ep == epoch:
                out[sig] = np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.float32)
        return out

    def clear_epoch(self, epoch: int | None) -> None:
        """Remove all buffers for a given epoch to free memory after flushing."""
        keys = [k for k in self._buffers if k[0] == epoch]
        for k in keys:
            self._buffers.pop(k, None)

    def flush_epoch_to_npz(self, epoch: int | None, out_dir: str | Path) -> Path:
        """
        Save all buffers for the given epoch into a single NPZ file.

        Returns:
            Path to the written NPZ.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        epoch_str = "none" if epoch is None else f"{epoch:04d}"
        out_path = out_dir / f"epoch_{epoch_str}.npz"

        data = self.get_epoch_buffers(epoch)
        np.savez_compressed(out_path, **data)
        return out_path
