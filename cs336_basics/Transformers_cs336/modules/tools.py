import debugpy
import torch
import numpy.typing as npt
import numpy as np
import os
from torch.nn import Module
from torch.optim import Optimizer
from typing import Union, BinaryIO, IO

def save_checkpoint(
    model: Module,
    optimizer: Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]
) -> None:
    """
    Saves a training checkpoint to the given output path or file-like object.

    Parameters:
    - model (torch.nn.Module): The model to save.
    - optimizer (torch.optim.Optimizer): The optimizer to save.
    - iteration (int): The current training iteration step.
    - out (str | os.PathLike | BinaryIO | IO[bytes]): The path or file-like object to save to.

    The checkpoint is saved as a dictionary with:
        - 'model_state_dict'
        - 'optimizer_state_dict'
        - 'iteration'
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: Module,
    optimizer: Optimizer
) -> int:
    """
    Loads a checkpoint from the given source and restores the model and optimizer states.

    Parameters:
    - src (str | os.PathLike | BinaryIO | IO[bytes]): The checkpoint path or file-like object.
    - model (torch.nn.Module): The model whose state will be restored.
    - optimizer (torch.optim.Optimizer): The optimizer whose state will be restored.

    Returns:
    - iteration (int): The iteration number saved in the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def wait_for_debugger(port: int = 5678, host: str = "localhost") -> None:
    """
    Wait for a debugger to attach on the given host and port using debugpy.

    Args:
        port (int): Port to listen on. Default is 5678.
        host (str): Host to bind to. Default is "localhost".
    """
    print(f"Waiting for debugger to attach on {host}:{port}...")
    debugpy.listen((host, port))
    debugpy.wait_for_client()
    print("Debugger attached.")


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """   
    # We want to sample sequences of length `context_length` from the dataset for each batch.
    max_start = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start + 1, size=batch_size)

    # Prepare slices in NumPy
    input_np = np.stack([
        dataset[i : i + context_length] for i in start_indices
    ])
    label_np = np.stack([
        dataset[i + 1 : i + context_length + 1] for i in start_indices
    ])

    # Convert to torch in one go
    input_tensor = torch.tensor(input_np, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(label_np, dtype=torch.long, device=device)

    return [input_tensor, labels_tensor]