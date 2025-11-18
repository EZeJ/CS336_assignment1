import os
import time
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
import llm_backbone.Transformers_cs336 as my_tf
import wandb
import math


torch.set_float32_matmul_precision("medium")


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train transformer LM")
    parser.add_argument(
        "--config",
        default="./llm_backbone/configures/m4.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # if we need to debug, we can wait for the debugger to attach
    # my_tf.modules.wait_for_debugger(port=5678, host="localhost")



    # Load config
    config = load_config(args.config)

    wandb_flag = config["training"]["wandb"]
    wandb_project = config["training"]["wandb_project"]
    tb_logdir = config["training"].get("tensorboard_logdir")
    writer = SummaryWriter(tb_logdir) if tb_logdir else None
    use_data_parallel = config["training"].get("data_parallel", False)

    if wandb_flag:
        config_name = Path(args.config).stem
        run_name = f"llm_training_{config_name}_{int(time.time())}"
        wandb.init(project=wandb_project, name=run_name, config=config)
    device = detect_device() if config["training"]["device"] == "auto" else config["training"]["device"]

    # Load dataset
    train_data = np.memmap(config["dataset"]["train_path"], dtype=np.uint16, mode="r")
    val_data = np.memmap(config["dataset"]["val_path"], dtype=np.uint16, mode="r")

    # max_l2_norm
    max_l2_norm = config["optimizer"]["max_l2_norm"]
    # ensure checkpoint directory exists
    ckpt_dir = os.path.dirname(config["training"]["checkpoint_path"])
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    
    # Create model
    model = my_tf.transformer.Transformer(
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        vocab_size=config["model"]["vocab_size"],
        context_length=config["model"]["context_length"],
        num_layers=config["model"]["num_layers"],
        max_seq_len=config["model"]["context_length"],
        theta=config["model"]["rope_theta"],
        device=device
    ).to(device)
    if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.train()

    # Create optimizer
    optimizer = my_tf.modules.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate_max"]),
        weight_decay=float(config["optimizer"]["weight_decay"])
    )

    # Treat max_iters as total number of epochs; single tqdm only
    max_epochs = config["training"]["max_iters"]
    warmup_iters = min(config["optimizer"]["warmup_iters"], max_epochs)
    cosine_iters = min(config["optimizer"]["cosine_iters"], max_epochs)
    cosine_iters = max(cosine_iters, warmup_iters + 1)

    last_val_loss = None
    epoch_bar = trange(max_epochs, desc="epoch", leave=True)

    for epoch in epoch_bar:
        # Update LR per epoch
        lr = my_tf.modules.get_lr_cosine_schedule(
            epoch,
            float(config["optimizer"]["learning_rate_max"]),
            float(config["optimizer"]["learning_rate_min"]),
            warmup_iters,
            cosine_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # One batch per epoch (as requested)
        x, y = my_tf.modules.get_batch(
            dataset=train_data,
            batch_size=config["training"]["batch_size"],
            context_length=config["model"]["context_length"],
            device=device
        )

        # Forward
        logits = model(x)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = y.reshape(-1)
        loss = my_tf.modules.get_cross_entropy_loss(logits_flat, targets_flat)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward() 
        my_tf.modules.get_gradient_clipping(model.parameters(), max_l2_norm=max_l2_norm)
        optimizer.step()

        if wandb_flag:
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "epoch": epoch})
        if writer:
            writer.add_scalar("train/loss", loss.item(), epoch)
            writer.add_scalar("train/lr", lr, epoch)

        # Validation
        if (epoch + 1) % config["training"]["val_every"] == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = my_tf.modules.get_batch(
                    dataset=val_data,
                    batch_size=config["training"]["batch_size"],
                    context_length=config["model"]["context_length"],
                    device=device
                )
                logits_val = model(x_val)
                val_loss = my_tf.modules.get_cross_entropy_loss(
                    logits_val.view(-1, logits_val.size(-1)),
                    y_val.reshape(-1)
                )
                last_val_loss = val_loss.item()
                
                if wandb_flag:
                    wandb.log({"val/loss": val_loss.item(), "epoch": epoch})
                if writer:
                    writer.add_scalar("val/loss", val_loss.item(), epoch)
                # print(f"[Validation] Epoch {epoch}: val_loss = {val_loss.item():.4f}")
            model.train()

        # Save checkpoint
        if (epoch + 1) % config["training"]["val_every"] == 0:
            my_tf.modules.save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=epoch,
                out=config["training"]["checkpoint_path"]
            )

        epoch_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            vloss=f"{last_val_loss:.4f}" if last_val_loss is not None else "n/a",
            lr=f"{lr:.6f}",
        )

    epoch_bar.close()

    if writer:
        writer.flush()
        writer.close()

if __name__ == "__main__":
    main()
