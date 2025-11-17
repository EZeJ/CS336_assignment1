import os
import time
import argparse
import yaml
import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import llm_backbone.Transformers_cs336 as my_tf
import wandb



torch.set_float32_matmul_precision("medium")


def detect_device() -> str:
    if torch.cuda.is_available():
        # pick the CUDA device with the most free memory to avoid busy GPUs
        best = None
        max_free = -1
        for idx in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(idx)
            if free > max_free:
                max_free = free
                best = idx
        return f"cuda:{best}" if best is not None else "cuda"
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

    wandb_flag = config["training"]['wandb']
    tb_logdir = config["training"].get("tensorboard_logdir")
    writer = SummaryWriter(tb_logdir) if tb_logdir else None

    if wandb_flag:
        wandb.init(project=config["training"]['wandb_project'], config=config)
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
    model.train()

    # Create optimizer
    optimizer = my_tf.modules.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate_max"]),
        weight_decay=float(config["optimizer"]["weight_decay"])
    )

    # Training loop
    progress = trange(config["training"]["max_iters"], desc="train")
    for it in progress:
        # Update LR
        lr = my_tf.modules.get_lr_cosine_schedule(
            it,
            float(config["optimizer"]["learning_rate_max"]),
            float(config["optimizer"]["learning_rate_min"]),
            config["optimizer"]["warmup_iters"],
            config["optimizer"]["cosine_iters"]
        )
        
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Get batch
        x, y = my_tf.modules.get_batch(
            dataset=train_data,
            batch_size=config["training"]["batch_size"],
            context_length=config["model"]["context_length"],
            device=device
        )

        # Forward
        logits = model(x)
        # Compute loss over all positions, not just the final token
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = y.reshape(-1)
        loss = my_tf.modules.get_cross_entropy_loss(logits_flat, targets_flat)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward() 
        my_tf.modules.get_gradient_clipping(model.parameters(), max_l2_norm=max_l2_norm)
        optimizer.step()

        # print(f"Step {it}: loss = {loss.item():.4f}, lr = {lr:.6f}")
        if wandb_flag:
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": it})
        if writer:
            writer.add_scalar("train/loss", loss.item(), it)
            writer.add_scalar("train/lr", lr, it)

        # Logging
        if it % config["training"]["log_every"] == 0:
            print(f"Step {it}: loss = {loss.item():.4f}, lr = {lr:.6f}")
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")

        # Validationls
        if it % config["training"]["val_every"] == 0 and it > 0:
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
                
                if wandb_flag:
                    wandb.log({"val/loss": val_loss.item(), "step": it})
                if writer:
                    writer.add_scalar("val/loss", val_loss.item(), it)
                print(f"[Validation] Step {it}: val_loss = {val_loss.item():.4f}")
            model.train()

        # Save checkpoint
        if it % config["training"]["val_every"] == 0:
            my_tf.modules.save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=it,
                out=config["training"]["checkpoint_path"]
            )

    if writer:
        writer.flush()
        writer.close()

if __name__ == "__main__":
    main()
