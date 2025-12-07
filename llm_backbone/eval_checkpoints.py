from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import llm_backbone.Transformers_cs336 as my_tf


torch.set_float32_matmul_precision("medium")


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def build_model(config: Dict[str, Any], device: str) -> torch.nn.Module:
    model_cfg = config["model"]
    model = my_tf.transformer.Transformer(
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        vocab_size=model_cfg["vocab_size"],
        context_length=model_cfg["context_length"],
        num_layers=model_cfg["num_layers"],
        max_seq_len=model_cfg["context_length"],
        theta=model_cfg["rope_theta"],
        device=device,
    )
    model.to(device)
    model.eval()
    return model


def load_memmap(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def evaluate_on_split(
    model: torch.nn.Module,
    dataset: np.memmap,
    config: Dict[str, Any],
    device: str,
    num_batches: int,
    batch_size: int,
) -> Dict[str, float]:
    context_length = config["model"]["context_length"]
    losses: List[float] = []
    perplexities: List[float] = []

    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            input_ids, target_ids = my_tf.modules.get_batch(
                dataset=dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logits = model(input_ids)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.reshape(-1)
            loss_tensor = my_tf.modules.get_cross_entropy_loss(logits_flat, targets_flat)
            ppl_tensor = torch.exp(loss_tensor)
            losses.append(float(loss_tensor.item()))
            perplexities.append(float(ppl_tensor.item()))

    return {
        "loss": float(np.mean(losses)),
        "perplexity": float(np.mean(perplexities)),
    }


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: str,
) -> Optional[int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model_state = model.state_dict()
    if set(state_dict.keys()) != set(model_state.keys()):
        stripped_state: Dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                stripped_state[key[len("module.") :]] = value
            else:
                stripped_state[key] = value
        state_dict = stripped_state

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"[warn] Incompatible keys for {checkpoint_path.name}: {asdict(incompatible)}")

    model.to(device)
    model.eval()
    return checkpoint.get("iteration")


def find_checkpoints(checkpoints_dir: Path) -> List[Path]:
    # Only consider checkpoints directly under the given directory (no subfolders).
    checkpoint_paths = sorted(checkpoints_dir.glob("*.pt"))
    return checkpoint_paths


def print_markdown_table(results: List[Dict[str, Any]], split: str) -> None:
    header = (
        f"| # | checkpoint | iteration | {split}_loss | {split}_perplexity |\n"
        f"|---|-----------|-----------|-----------:|-----------------:|"
    )
    print(header)
    for index, entry in enumerate(results):
        name = entry["name"]
        iteration = entry.get("iteration")
        loss_value = entry[f"{split}_loss"]
        perplexity_value = entry[f"{split}_perplexity"]
        print(
            f"| {index} | {name} | {iteration if iteration is not None else 'n/a'} | "
            f"{loss_value:.4f} | {perplexity_value:.4f} |"
        )


def plot_metrics(
    results: List[Dict[str, Any]],
    output_dir: Path,
    split: str,
) -> None:
    if not results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(
        results,
        key=lambda entry: (
            entry.get("iteration") if entry.get("iteration") is not None else float("inf"),
            entry["name"],
        ),
    )

    x_values: List[float] = []
    labels: List[str] = []
    losses: List[float] = []
    perplexities: List[float] = []

    for index, entry in enumerate(sorted_results):
        iteration = entry.get("iteration")
        x_values.append(float(iteration) if iteration is not None else float(index))
        labels.append(entry["name"])
        losses.append(entry[f"{split}_loss"])
        perplexities.append(entry[f"{split}_perplexity"])

    figure, axis_loss = plt.subplots(figsize=(8, 5))
    axis_ppl = axis_loss.twinx()

    axis_loss.plot(x_values, losses, marker="o", color="tab:blue", label=f"{split} loss")
    axis_ppl.plot(x_values, perplexities, marker="s", color="tab:orange", label=f"{split} perplexity")

    axis_loss.set_xlabel("iteration (or index)")
    axis_loss.set_ylabel("cross-entropy loss", color="tab:blue")
    axis_ppl.set_ylabel("perplexity", color="tab:orange")

    axis_loss.tick_params(axis="y", labelcolor="tab:blue")
    axis_ppl.tick_params(axis="y", labelcolor="tab:orange")

    figure.suptitle(f"Checkpoint {split} metrics")
    figure.tight_layout()
    figure.savefig(output_dir / f"checkpoint_{split}_metrics.png", dpi=200)
    plt.close(figure)

    figure_bar, axis_bar = plt.subplots(figsize=(10, 4))
    positions = np.arange(len(sorted_results))
    axis_bar.bar(positions, perplexities, color="tab:orange")
    axis_bar.set_xticks(positions)
    axis_bar.set_xticklabels(labels, rotation=45, ha="right")
    axis_bar.set_ylabel(f"{split} perplexity")
    axis_bar.set_title(f"Checkpoint {split} perplexity by file")
    figure_bar.tight_layout()
    figure_bar.savefig(output_dir / f"checkpoint_{split}_perplexity_by_file.png", dpi=200)
    plt.close(figure_bar)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate language model checkpoints on cross-entropy and perplexity.")
    parser.add_argument(
        "--config",
        type=str,
        default="./llm_backbone/configures/best_val.yaml",
        help="Path to YAML config describing model and dataset.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="./llm_backbone/checkpoints",
        help="Directory containing .pt checkpoints to evaluate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="Which dataset split to evaluate on.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Number of random batches to sample for each checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size to use; defaults to value from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use (e.g. "cpu", "cuda"); "auto" picks a reasonable default.',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots and tables; defaults to <checkpoints-dir>/analysis.",
    )
    arguments = parser.parse_args()

    config_path = Path(arguments.config)
    checkpoints_dir = Path(arguments.checkpoints_dir)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    config = load_config(config_path)
    training_cfg = config["training"]
    batch_size = arguments.batch_size or int(training_cfg["batch_size"])

    device = detect_device() if arguments.device == "auto" else arguments.device
    print(f"Using device: {device}")

    if arguments.split == "train":
        data_path = Path(config["dataset"]["train_path"])
    else:
        data_path = Path(config["dataset"]["val_path"])

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found for split {arguments.split}: {data_path}")

    dataset = load_memmap(data_path)
    model = build_model(config, device=device)
    checkpoint_paths = find_checkpoints(checkpoints_dir)

    if not checkpoint_paths:
        print(f"No .pt checkpoints found under {checkpoints_dir}")
        return

    print(f"Found {len(checkpoint_paths)} checkpoints under {checkpoints_dir}")

    results: List[Dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating checkpoint: {checkpoint_path}")
        iteration = load_checkpoint_into_model(model, checkpoint_path, device=device)
        metrics = evaluate_on_split(
            model=model,
            dataset=dataset,
            config=config,
            device=device,
            num_batches=arguments.num_batches,
            batch_size=batch_size,
        )
        results.append(
            {
                "name": checkpoint_path.name,
                "path": str(checkpoint_path),
                "iteration": iteration,
                f"{arguments.split}_loss": metrics["loss"],
                f"{arguments.split}_perplexity": metrics["perplexity"],
            }
        )

    output_dir = Path(arguments.output_dir) if arguments.output_dir is not None else checkpoints_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / f"checkpoint_{arguments.split}_metrics.md"
    with markdown_path.open("w") as handle:
        original_stdout = handle
        header = (
            f"| # | checkpoint | iteration | {arguments.split}_loss | {arguments.split}_perplexity |\n"
            f"|---|-----------|-----------|-----------:|-----------------:|\n"
        )
        handle.write(header)
        for index, entry in enumerate(results):
            iteration_value = entry.get("iteration")
            handle.write(
                f"| {index} | {entry['name']} | "
                f"{iteration_value if iteration_value is not None else 'n/a'} | "
                f"{entry[f'{arguments.split}_loss']:.4f} | "
                f"{entry[f'{arguments.split}_perplexity']:.4f} |\n"
            )

    print_markdown_table(results, split=arguments.split)
    plot_metrics(results, output_dir=output_dir, split=arguments.split)
    print(f"Saved table and plots to {output_dir}")


if __name__ == "__main__":
    main()
