import torch 
from torch import Tensor
import math
from einops import rearrange, einsum
from torch.nn.parameter import Parameter



class Embedding(torch.nn.Module):
    r"""
    Construct an embedding module.

    Args:
        num_embeddings (int): Size of the vocabulary. Each index from 0 to num_embeddings - 1 
            will have a corresponding embedding vector.
        embedding_dim (int): Dimension of each embedding vector (i.e., d_model).
        device (torch.device | None, optional): The device on which to store the embedding 
            parameters. Defaults to None, which lets PyTorch choose the default device.
        dtype (torch.dtype | None, optional): The data type of the embedding parameters. 
            Defaults to None.

    Example:
        >>> emb = MyEmbedding(num_embeddings=1000, embedding_dim=512, device=torch.device("cuda"))
        >>> x = torch.tensor([1, 5, 9])
        >>> output = emb(x)  # output shape: (3, 512)
    """
    def __init__(
            self, 
            num_embeddings:int,
            embedding_dim: int,
            device:torch.device = None,
            dtype:torch.dtype = None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__() 
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        