"""
Training script variant with activation logging for GP datasets.

Uses SwiGLUWithLogging and RMSNormWithLogging to capture gate and RMS signals.
"""

from __future__ import annotations

import os
import argparse
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import llm_backbone.Transformers_cs336 as my_tf
from GP.logger_GP import ActivationLogger
from llm_backbone.Transformers_cs336.modules.SwiGLU_GP import SwiGLUWithLogging
from llm_backbone.Transformers_cs336.modules.RMSNorm_GP import RMSNormWithLogging


torch.set_float32_matmul_precision("medium")


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(config: dict, device: str, logger: ActivationLogger | None, epoch_ref: list[int]):
    """
    Build a Transformer model that uses logging-capable SwiGLU and RMSNorm modules.
    epoch_ref: a single-element list holding current epoch for logger context.
    """

    class TransformerWithLogging(my_tf.transformer.Transformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # swap modules in-place
            for i, block in enumerate(self.transformer_layers):
                block.SwiGLU_ffn = SwiGLUWithLogging(
                    d_model=config["model"]["d_model"],
                    d_ff=config["model"]["d_ff"],
                    device=device,
                    logger=logger,
                    log_prefix=f"layer{i}_",
                )
                block.RMSNorm_ln1 = RMSNormWithLogging(
                    d_model=config["model"]["d_model"],
                    device=device,
                    logger=logger,
                    log_prefix=f"layer{i}_",
                )
                block.RMSNorM_ln2 = RMSNormWithLogging(
                    d_model=config["model"]["d_model"],
                    device=device,
                    logger=logger,
                    log_prefix=f"layer{i}_",
                )

        def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
            # Override to pass epoch into blocks (for logging keys)
            x = self.embedding(in_indices)
            batch_size, seq_len = in_indices.shape
            pos_ids = torch.arange(0, seq_len, device=in_indices.device)
            for block in self.transformer_layers:
                x = block(x, token_positions=pos_ids, epoch=epoch_ref[0])
            x = self.RMSNorm_ln_final(x)
            logits = self.lm_head(x)
            return logits

    model = TransformerWithLogging(
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        vocab_size=config["model"]["vocab_size"],
        context_length=config["model"]["context_length"],
        num_layers=config["model"]["num_layers"],
        max_seq_len=config["model"]["context_length"],
        theta=config["model"]["rope_theta"],
        device=device,
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train transformer with activation logging for GP.")
    parser.add_argument("--config", default="./llm_backbone/configures/valid.yaml", help="Path to YAML config file.")
    parser.add_argument("--log-dir", default="./GP/datasets/raw", help="Output directory for NPZ logs.")
    parser.add_argument("--max-samples", type=int, default=10000, help="Max samples per logging call.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for logging subsampling.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = detect_device() if config["training"]["device"] == "auto" else config["training"]["device"]

    # Data
    train_data = np.memmap(config["dataset"]["train_path"], dtype=np.uint16, mode="r")
    val_data = np.memmap(config["dataset"]["val_path"], dtype=np.uint16, mode="r")

    # Logger
    act_logger = ActivationLogger(max_samples_per_call=args.max_samples, rng_seed=args.seed)
    epoch_ref = [0]  # mutable holder for current epoch value

    # Model
    model = build_model(config, device=device, logger=act_logger, epoch_ref=epoch_ref)
    use_data_parallel = config["training"].get("data_parallel", False)
    if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.train()

    # Optimizer and schedules
    optimizer = my_tf.modules.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate_max"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    max_epochs = config["training"]["max_iters"]
    warmup_iters = min(config["optimizer"]["warmup_iters"], max_epochs)
    cosine_iters = max(min(config["optimizer"]["cosine_iters"], max_epochs), warmup_iters + 1)
    max_l2_norm = config["optimizer"]["max_l2_norm"]
    val_every = config["training"]["val_every"]

    writer = SummaryWriter(config["training"].get("tensorboard_logdir")) if config["training"].get("tensorboard_logdir") else None

    epoch_bar = trange(max_epochs, desc="epoch", leave=True)

    for epoch in epoch_bar:
        epoch_ref[0] = epoch
        lr = my_tf.modules.get_lr_cosine_schedule(
            epoch,
            float(config["optimizer"]["learning_rate_max"]),
            float(config["optimizer"]["learning_rate_min"]),
            warmup_iters,
            cosine_iters,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = my_tf.modules.get_batch(
            dataset=train_data,
            batch_size=config["training"]["batch_size"],
            context_length=config["model"]["context_length"],
            device=device,
        )
        logits = model(x)
        loss = my_tf.modules.get_cross_entropy_loss(logits.view(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        my_tf.modules.get_gradient_clipping(model.parameters(), max_l2_norm=max_l2_norm)
        optimizer.step()

        if writer:
            writer.add_scalar("train/loss", loss.item(), epoch)
            writer.add_scalar("train/lr", lr, epoch)

        # Validation and checkpoint (optional)
        if (epoch + 1) % val_every == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = my_tf.modules.get_batch(
                    dataset=val_data,
                    batch_size=config["training"]["batch_size"],
                    context_length=config["model"]["context_length"],
                    device=device,
                )
                logits_val = model(x_val)
                val_loss = my_tf.modules.get_cross_entropy_loss(
                    logits_val.view(-1, logits_val.size(-1)),
                    y_val.reshape(-1),
                )
                if writer:
                    writer.add_scalar("val/loss", val_loss.item(), epoch)
            model.train()
            # optional checkpoint could be added here if desired

        # Flush logs per epoch
        out_path = act_logger.flush_epoch_to_npz(epoch, args.log_dir)
        act_logger.clear_epoch(epoch)
        epoch_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}", log=str(out_path))

    if writer:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
