import math
from typing import Optional, List
import torch
from torch import nn


class PrepareForMultiHeadAttention(nn.Module):
    """
    A module for preparing input tensors for multi-head attention mechanism.

    Args:
        d_model (int): The dimension of the input feature representation.
        heads (int): The number of attention heads.
        d_k (int): The dimension of each attention head.
        bias (bool): If True, a bias term is added to the linear transformation.

    Attributes:
        linear (nn.Linear): Linear transformation layer to project input to (num_heads * d_k) dimension.
        heads (int): The number of attention heads.
        d_k (int): The dimension of each attention head.

    Note:
        This module is typically used as a preparatory step before applying multi-head attention.

    Shape:
        - Input: (batch_size, sequence_length, d_model)
        - Output: (batch_size, sequence_length, num_heads, d_k)

    Example:
        >>> d_model = 512
        >>> num_heads = 8
        >>> d_k = 64
        >>> bias = True
        >>> prepare = PrepareForMultiHeadAttention(d_model, num_heads, d_k, bias)
        >>> input_tensor = torch.randn(16, 20, d_model)
        >>> output_tensor = prepare(input_tensor)
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PrepareForMultiHeadAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, sequence_length, num_heads, d_k).
        """
        head_shape = x[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True
    ):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum("ibhd,jbhd->ijbh", query, key)

    def prepare_mask(
        self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]
    ):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        mask = mask.unsqueeze(-1)
        return mask

    def forward(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
            query = self.query(query)
            key = self.key(key)
            value = self.value(value)
            scores = self.get_scores(query, key)
            scores *= self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attn = self.softmax(scores)
            attn = self.softmax(scores)
            attn = self.dropout(attn)
            x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
            self.attn = attn.detach()
            x = x.reshape(seq_len, batch_size, -1)
            return self.output(x)