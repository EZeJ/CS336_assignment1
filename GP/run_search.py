"""
Run GP search over activations_all.npz and save checkpoints/results.

Usage:
    uv run python -m GP.run_search --config ./GP/configs/default.yaml --npz ./GP/datasets/raw/20251116_210338/activations_all.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from dataclasses import asdict
from GP.utils import load_yaml_config, load_run_yaml, make_run_dir, save_checkpoint
from GP.datasets import Dataset
from GP.gp import GPSearch
from GP.expr import Expr
from GP.aggregate_GP import TARGET_FUNCS


def _resolve_target_func(sig: str):
    """
    Map signal names (possibly prefixed with layer info) to target functions.
    Falls back to identity.
    """
    if sig in TARGET_FUNCS:
        return TARGET_FUNCS[sig]

    for base in TARGET_FUNCS:
        if sig.endswith(base):
            return TARGET_FUNCS[base]
    return lambda z: z


def build_dataset_from_signal(npz: dict, sig: str) -> Dataset:
    x = np.array(npz[sig]).astype(np.float32)
    fn = _resolve_target_func(sig)
    y = fn(x).astype(np.float32)
    return Dataset(inputs=x[:, None], targets=y, feature_names=["x"])


def collect_signals(npz: dict, target_key: str) -> list[str]:
    """
    Collect all signals whose name ends with target_key.
    For example, target_key='rms_mean_sq' will gather all layer-prefixed variants.
    """
    return [k for k in npz.files if k.endswith(target_key)]


def build_combined_dataset(npz: dict, sigs: list[str], name: str) -> Dataset:
    xs = []
    ys = []
    for sig in sigs:
        x = np.array(npz[sig]).astype(np.float32)
        fn = _resolve_target_func(sig)
        y = fn(x).astype(np.float32)
        xs.append(x)
        ys.append(y)
    x_all = np.concatenate(xs).astype(np.float32)
    y_all = np.concatenate(ys).astype(np.float32)
    return Dataset(inputs=x_all[:, None], targets=y_all, feature_names=["x"])


def main():
    parser = argparse.ArgumentParser(description="Run GP search on activations NPZ.")
    parser.add_argument("--config", help="YAML config path for GP hyperparameters.")
    parser.add_argument("--train-config", help="YAML run-level config (npz, combine, jobs, etc.).")
    parser.add_argument("--npz", help="Path to activations_all.npz.")
    parser.add_argument("--signals", nargs="*", default=None, help="Signals to fit; default = all non-epoch keys.")
    parser.add_argument("--target-key", default=None, help="If set, train on all signals ending with this key (e.g., rms_mean_sq).")
    parser.add_argument("--out-dir", default="./GP/checkpoints", help="Base checkpoints directory.")
    parser.add_argument("--verbose", action="store_true", help="Print per-generation best during search.")
    parser.add_argument("--log-every", type=int, default=10, help="Generations between verbose logs.")
    parser.add_argument("--combine", action="store_true", help="Combine all signals matching target-key into one dataset.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel worker processes for fitness eval.")
    args = parser.parse_args()

    run_cfg = load_run_yaml(args.train_config) if args.train_config else {}

    # Determine GP config path: train-config gp_config overrides CLI --config
    gp_cfg_path = run_cfg.get("gp_config", args.config)
    if gp_cfg_path is None:
        raise ValueError("Must provide either --config or train-config with gp_config field.")

    cfg = load_yaml_config(gp_cfg_path)

    # Determine NPZ path: train-config npz overrides CLI --npz
    npz_path_str = run_cfg.get("npz", args.npz)
    if npz_path_str is None:
        raise ValueError("Must provide NPZ path via --npz or train-config npz field.")
    npz_path = Path(npz_path_str)
    data = np.load(npz_path)
    # target key: CLI > config > None
    target_key = args.target_key or run_cfg.get("target_key") or cfg.target_key
    if target_key:
        signals = collect_signals(data, target_key)
    else:
        signals = run_cfg.get("signals") or args.signals or [k for k in data.files if not k.endswith("_epoch")]

    out_dir = run_cfg.get("out_dir", args.out_dir)
    run_dir = make_run_dir(out_dir, prefix="gp")
    # Save config snapshot
    save_checkpoint(run_dir, "config", asdict(cfg))

    # Setup logging with wandb/weave
    wandb_run = None
    run_name_base = Path(args.train_config or gp_cfg_path).stem
    run_name = f"{run_name_base}_{run_dir.name}"
    try:
        import wandb

        try:
            import weave

            weave.init("680")
        except Exception:
            pass

        wandb_run = wandb.init(
            project="680",
            name=run_name,
            config={
                "gp_config": str(gp_cfg_path),
                "npz": str(npz_path),
                "target_key": target_key,
                "combine": run_cfg.get("combine", args.combine),
                "jobs": int(run_cfg.get("jobs", args.jobs)),
            },
        )
    except Exception as e:
        print(f"[warn] wandb/weave logging disabled: {e}")
        wandb_run = None

    # Print config and signals upfront
    print("=== GP config ===")
    print(cfg)
    print("Signals:", signals)

    results = []
    combine = run_cfg.get("combine", args.combine)
    jobs = int(run_cfg.get("jobs", args.jobs))
    verbose = bool(run_cfg.get("verbose", args.verbose))
    log_every = int(run_cfg.get("log_every", args.log_every))

    def make_logger():
        if wandb_run is None:
            return None

        import wandb

        def _log(gen, gen_best, best_overall):
            m = gen_best.metrics
            gm = best_overall.metrics
            wandb.log(
                {
                    "gen": gen,
                    "best/loss": m.loss,
                    "best/mean_rel": m.mean_rel,
                    "best/max_rel": m.max_rel,
                    "best/degree": m.degree,
                    "best/multiplies": m.multiplies,
                    "best/depth": m.depth,
                    "best/term_count": m.term_count,
                    "best/expr": str(gen_best.expr),
                    "global/best_loss": gm.loss,
                    "global/mean_rel": gm.mean_rel,
                    "global/max_rel": gm.max_rel,
                    "global/degree": gm.degree,
                    "global/multiplies": gm.multiplies,
                    "global/depth": gm.depth,
                    "global/term_count": gm.term_count,
                    "global/expr": str(best_overall.expr),
                },
                step=gen,
            )

        return _log

    logger = make_logger()

    if target_key and combine:
        ds = build_combined_dataset(data, signals, target_key)
        search = GPSearch(cfg, ds, verbose=verbose, log_every=log_every, n_jobs=jobs, logger=logger)
        best, _ = search.run()
        res = {
            "signal": f"combined_{target_key}",
            "expression": str(best.expr),
            "metrics": asdict(best.metrics),
        }
        results.append(res)
        save_checkpoint(run_dir, f"best_combined_{target_key}", res)
        print(f"[combined {target_key}] best expr: {res['expression']}, metrics: {res['metrics']}")
    else:
        for sig in signals:
            ds = build_dataset_from_signal(data, sig)
            search = GPSearch(cfg, ds, verbose=verbose, log_every=log_every, n_jobs=jobs, logger=logger)
            best, _ = search.run()
            res = {
                "signal": sig,
                "expression": str(best.expr),
                "metrics": asdict(best.metrics),
            }
            results.append(res)
            save_checkpoint(run_dir, f"best_{sig}", res)
            print(f"[{sig}] best expr: {res['expression']}, metrics: {res['metrics']}")

    # Save aggregate results
    save_checkpoint(run_dir, "results", {"signals": results, "npz": str(npz_path), "config": asdict(cfg)})

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
