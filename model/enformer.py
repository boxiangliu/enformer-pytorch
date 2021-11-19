import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from model.attention import MultiHeadAttention
import numpy as np

SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896

class Print(nn.Module):
    def __init__(self, name):
        super(Print, self).__init__()
        self._name = name

    def forward(self, x):
        print(f"{self._name}: {x.shape}")
        return x

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
            # b: batch
            # l: length
            # c: channel
            Print(1),
            Rearrange("b l c -> b c l"),
            Print(2),
            nn.Conv1d(num_alphabet, channels // 2, 15, padding="same"),
            Print(3),
            Residual(conv_block(channels // 2, channels // 2, 1)),
            Print(4),
            SoftmaxPooling1D(channels // 2, pool_size=2)
        )

        filter_list = exponential_linspace_int(
            channels // 2, channels, num=6, divisible_by=128)
        filter_list = [channels // 2, *filter_list]

        conv_layers = []
        for in_channels, out_channels in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                nn.Sequential(
                    conv_block(in_channels, out_channels, 5),
                    Residual(conv_block(out_channels, out_channels, 1)),
                    SoftmaxPooling1D(out_channels, pool_size=2),
                    Print("conv tower")
                )
            )
        conv_tower = nn.Sequential(*conv_layers)

        attn_kwargs = {
            "input_dim": channels,
            "value_dim": channels // num_heads,
            "key_dim": 64,
            "num_heads": num_heads,
            "scaling": True,
            "attention_dropout_rate": 0.05,
            "relative_position_symmetric": False,
            "num_relative_position_features": channels // num_heads,
            "positional_dropout_rate": 0.01,
            "zero_initialize": True
        }

        def transformer_mlp():
            return Residual(nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels * 2),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(channels * 2, channels),
                nn.Dropout(dropout_rate)
            ))

        transformer = []
        for _ in range(num_transformer_layers):
            transformer.append(
                nn.Sequential(
                    Residual(nn.Sequential(
                        nn.LayerNorm(channels),
                        MultiHeadAttention(**attn_kwargs),
                        nn.Dropout(dropout_rate),
                        Print("transformer")
                    )),
                    transformer_mlp(),
                    Print("MLP")
                )
            )

        transformer = nn.Sequential(
            Rearrange("b c l -> b l c"),
            Print("before transformer"),
            *transformer
        )

        crop_final = TargetLengthCrop(TARGET_LENGTH)
        final_pointwise = nn.Sequential(
            nn.Linear(channels, channels * 2, 1),
            nn.Dropout(dropout_rate / 8),
            GELU()
        )

        self._trunk = nn.Sequential(
            stem,
            conv_tower,
            transformer,
            crop_final,
            Print("crop"),
            final_pointwise,
            Print("final")
        )

        self._heads = nn.ModuleDict({
            head: nn.Sequential(
                nn.Linear(channels * 2, head_channels, 1),
                nn.Softplus())
            for head, head_channels in heads_channels.items()
        })

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    def forward(self, inputs):
        x = self.trunk(inputs)
        return {head: head_module(x) for
                head, head_module in self.heads.items()}


class TargetLengthCrop(nn.Module):

    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2
        return x[:, -trim:trim, :]


class Residual(nn.Module):
    """residuel block"""

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, x, *args, **kwargs):
        return x + self._module(x, *args, **kwargs)


def conv_block(in_channels, out_channels, kernel_size=1, **kwargs):
    return nn.Sequential(
        nn.BatchNorm1d(in_channels),
        GELU(),
        nn.Conv1d(in_channels, out_channels,
                  kernel_size, padding="same", **kwargs)
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
        self._w_init_scale = w_init_scale
        self._logit_linear = nn.Linear(channels, channels, bias=False)
        self._logit_linear.weight.data.copy_(
            torch.eye(channels) * self._w_init_scale)

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, ("input length must "
                                              "by divisible by pool_size")
        x = rearrange(x, "b c (l p) -> b l p c", p=self._pool_size)
        x = x * F.softmax(self._logit_linear(x), dim=-2)
        x = torch.sum(x, dim=-2)
        return rearrange(x, "b l c -> b c l")


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value = 0,
                   dtype=np.float32) -> np.ndarray:
    """One-hot encode sequence."""
    def to_uint8(string):
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]
