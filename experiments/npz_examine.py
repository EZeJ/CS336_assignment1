from pathlib import Path
import numpy as np
from collections import defaultdict

# Path to aggregated NPZ from a logging run
# Example: Path("./GP/datasets/raw/20250101_120000/activations_all.npz")
# npz_path = Path("../GP/datasets/raw/20251116_210338/activations_all.npz")

project_root = Path(__file__).resolve().parent.parent  # go up one level from this file
npz_path = project_root / "GP" / "datasets" / "raw" / "20251116_210338" / "activations_all.npz"

assert npz_path.exists(), f"NPZ not found: {npz_path}"

data = np.load(npz_path)
signals = [k for k in data.files if not k.endswith("_epoch")]
print(f"Loaded signals: {signals}")


def summarize_array(arr: np.ndarray, name: str):
    arr = arr.astype(np.float32)
    stats = {
        "count": arr.size,
        "min": float(np.min(arr)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
        "mean": float(np.mean(arr)) if arr.size else None,
        "std": float(np.std(arr)) if arr.size else None,
        "p1": float(np.percentile(arr, 1)) if arr.size else None,
        "p50": float(np.percentile(arr, 50)) if arr.size else None,
        "p99": float(np.percentile(arr, 99)) if arr.size else None,
    }
    print(f"\n{name} summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

def per_epoch_counts(sig: str):
    ep_key = f"{sig}_epoch"
    if ep_key not in data:
        print(f"No epoch info for {sig}")
        return
    epochs = data[ep_key]
    counts = defaultdict(int)
    for ep in epochs:
        counts[int(ep)] += 1
    print(f"Per-epoch counts for {sig}:")
    for ep, cnt in sorted(counts.items()):
        print(f"  epoch {ep}: {cnt}")

def quick_hist(arr: np.ndarray, bins=10):
    if arr.size == 0:
        return []
    hist, edges = np.histogram(arr, bins=bins)
    return list(zip(hist.tolist(), edges[:-1].tolist(), edges[1:].tolist()))

for sig in signals:
    arr = data[sig]
    summarize_array(arr, sig)
    per_epoch_counts(sig)
    h = quick_hist(arr, bins=8)
    print("  hist (count, low, high):")
    for c, lo, hi in h:
        print(f"    {c:6d} in [{lo: .3f}, {hi: .3f})")
