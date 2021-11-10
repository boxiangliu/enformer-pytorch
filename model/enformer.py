import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, reduce


class Enformer(nn.Module):
    """Main class"""

    def __init__(self,
                 channels=1536,
                 num_heads=8,
                 num_transformer_layers=11,
                 pooling_type="attention"):
    """
    Args:
        channels: number of convolutional filters
        num_heads: number of attention heads
        num_transformer_layers: number of transformer layers
        pooling_type: "attention" or "max"
    """

    super().__init__()

    heads_channels = {"human": 5313, "mouse": 1643}
    dropout_rate = 0.4
    num_alphabet = 4
    assert channels % num_heads == 0, ("channels need to be "
                                       "divisible by heads")

    stem = nn.Sequential(
        # n: batch
        # l: length
        # c: channel
        Rearrange("n l c -> n c l")
        nn.Conv1d(num_alphabet, channels // 2, 15, padding="same"),
        Residual(conv_block(channels // 2, channels // 2, 1)),
        SoftmaxPooling1D(channels // 2, pool_size=2)
    )

    whole_attention_kwargs = {
        "attention_dropout_rate": 0.05,
        "initializer": None,
        "key_size": 64,
        "num_heads": num_heads,
        "num_relative_position_features": channels // num_heads,
        "positional_dropout_rate": 0.01,
        "relative_position_functions": [
            "positional_features_exponential",
            "positional_features_central_mask",
            "positional_features_gamma"
        ],
        "relative_positions": True,
        "scaling": True,
        "value_size": channels // num_heads,
        "zero_initialize": True
    }


class Residual(nn.Module):
    """residuel block"""

    def __init__(self, module):
        super().__init()
        self._module = module

    def forward(self, x, *args, **kwargs):
        return x + self._module(x, *args, **kwargs)


def conv_block(in_channels, out_channels, kernel_size=1, **kwargs):
    return nn.Sequential(
        nn.BatchNorm1d(in_channels),
        gelu,
        nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    )


class SoftmaxPooling1D(nn.Module):

    def __init__(self, channels, pool_size=2, w_init_scale=2.0):
        """
        Args:
            channels: number of channels
            pool_size: pooling size
            w_init_scale: scale on the diagonal element.
        """
        super().__init__()
        self._pool_size = pool_size
        self._per_channel = per_channel
        self._w_init_scale = w_init_scale
        self._logit_linear = nn.Linear(channels, channels, bias=False)
        self._logit_linear.weight.data.copy_(
            torch.eye(channels) * self._w_init_scale)

    def forward(self, x):
        assert x.shape[-1] % pool_size == 0, ("input length must "
                                              "by divisible by pool_size")
        x = rearrange(x, "n c (l p) -> n l p c", p=pool_size)
        x = x * F.softmax(self._logit_linear(x), axis=-2)
        x = torch.sum(x, dim=-2)
        return rearrange(x, "n l c -> n c l")


def gelu(x):
    return torch.sigmoid(1.702 * x) * x
