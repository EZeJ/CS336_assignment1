import torch
import torch.nn as nn
import torch.optim as optim
import time

def run_heavy_training(device="cuda", dtype=torch.float32):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(device)
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    # free_mem = free_mem / 4
    print(f"Free VRAM: {free_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB")

    input_dim = 2048
    output_dim = 2048
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    bytes_per_sample = (input_dim + output_dim) * bytes_per_element
    usable_bytes = int(free_mem * 0.25)  # Use 90% of available VRAM
    max_samples = usable_bytes // bytes_per_sample

    print(f"Preparing data on {device}...")
    print(f"Free VRAM → fitting ~{max_samples} samples of dim {input_dim} safely")

    X = torch.randn((max_samples, input_dim), dtype=dtype, device=device)
    Y = torch.randn((max_samples, output_dim), dtype=dtype, device=device)

    model = nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("🚀 Running dummy training...")
    for epoch in range(5000):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
        time.sleep(1)

if __name__ == "__main__":
    run_heavy_training()