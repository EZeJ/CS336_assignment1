import torch 
from torch import Tensor
import math
from einops import rearrange, einsum
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init

class Embedding(torch.nn.Module):
    r"""
    Construct an embedding module.

    Args:
        num_embeddings (int): Size of the vocabulary. Each index from 0 to num_embeddings - 1 
            will have a corresponding embedding vector (i.e., vocab_size). 
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
        self.weight = Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        self.reset_parameters()
    # End of the constructor

    def reset_parameters(self) -> None:
        init.trunc_normal_(
            self.weight, 
            mean = 0,
            std = 1,
            a = -3,
            b = 3
        )
    # End of the reset_parameters method

    def forward(self, token_ids: Tensor) -> Tensor:
        # test for the class structures
        # return F.embedding(token_ids, self.weight)
        return self.weight[token_ids]
    