import numpy as np
import torch
from torch import nn, einsum


class MultiHeadAttention(nn.Module):
    """multi-head attention"""

    def __init__(self,
                 input_dim,
                 value_dim,
                 key_dim,
                 num_heads,
                 scaling=True,
                 attention_dropout_rate=0.1,
                 relative_position_symmetric=False,
                 num_relative_position_features=None,
                 positional_dropout_rate=0.1,
                 zero_initialize=True):
        """Args:
            input_dim: the dimension of input embedding
            value_dim: the size of value embedding
            key_dim: the size of key embedding
            num_heads: the number of attention heads
            scaling: whether to scale the attention logits
            attention_dropout_rates: dropout rate for attention logits
            attention relative_position_symmetric: if True, the symmetric
            version of basis function will be used. if False, a symmeetric and
                asymmetric versions will be used.
            num_relative_position_features: number of relative positional
                features to compute. if None, `value_size * num_heads` is used.
            positional_dropout_rate: dropout rate for the positional encodings
                if relative positions are used
            zero_initialize: if True, the final linear layer will be 0 initialized
        """
        self._input_dim = input_dim
        self._value_dim = value_dim
        self._key_dim = key_dim
        self._num_heads = num_heads
        self._scaling = scaling
        self._attention_dropout_rate = attention_dropout_rate
        self._relative_positions = relative_positions
        self._relative_position_symmetric = relative_position_symmetric
        self._relative_position_functions = relative_position_functions
        if num_relative_position_features is None:
            divisible_by = 2 * len(self._relative_position_functions)
            self._num_relative_position_features = (
                (self._value_size // divisible_by) * divisible_by)
        else:
            self._num_relative_position_features = num_relative_position_features
        self._positional_dropout_rate = positional_dropout_rate

        key_proj_size = self._key_size * self._num_heads
        embedding_size = self._value_size * self._num_heads

        # query, key, and value weights
        self._q_layer = nn.Linear(input_dim, key_proj_size, bias=False)
        self._k_layer = nn.Linear(input_dim, key_proj_size, bias=False)
        self._v_layer = nn.Linear(input_dim, embedding_size, bias=False)
        self._embedding_layer = nn.Linear(embedding_size, input_dim)
        nn.init.zeros_(self._embedding_layer.weight)
        nn.init.zeros_(self._embedding_layer.bias)

        # relative position weights
        self._rel_pos_layer = nn.Linear(self._num_relative_position_features,
                                        key_proj_size, bias=False)
        self._rel_content_bias = nn.Parameter(
            torch.randn(1, self._num_heads, 1, self._key_dim))
        self._rel_pos_bias = nn.Parameter(
            torch.randn(1, self._num_heads, 1, self._key_dim))
        nn.init.kaiming_normal_(self._rel_content_bias)
        nn.init.kaiming_normal_(self._rel_pos_bias)

        # dropout layers:
        self._pos_dropout_layer = nn.Dropout(self._positional_dropout_rate)
        self._attn_dropout_layer = nn.Dropout(self._attention_dropout_rate)

    def forward(self, inputs):
        embedding_size = self._value_size * self._num_heads
        seq_len = inputs.shape[-2]

        q = self._q_layer(inputs)
        k = self._k_layer(inputs)
        v = self._v_layer(inputs)
        """
        b: batch
        h: head
        c: channel
        l: length
        """
        q, k, v = map(lambda x: rearrange(
            x, "b l (h c) -> b h l c", h=self.heads), (q, k, v))

        if self._scaling:
            q *= self._key_dim ** -0.5

        distances = torch.arange(-seq_len + 1, seq_len, device=inputs.device)
        positional_encodings = positional_features_all(
            positions=distances,
            feature_size=self._num_relative_position_features,
            seq_length=seq_len,
            symmetric=True)
        # [Batch, 2L - 1, Cr]

        positional_encodings = self._pos_dropout_layer(positional_encodings)
        rel_k = self._rel_pos_layer(positional_encodings)
        rel_k = rearrange("l (h c) -> h l c", rel_k)

        rel_logits = einsum("b h i c, h j c -> b h i j",
                            q + self._rel_pos_bias, rel_k)  # [B, H, L, 2L-1]
        rel_logits = relative_shift(rel_logits)  # [B, H, L, L]

        content_logits = einsum("b h i c, b h j c -> b h i j",
                                q + self._rel_content_bias, k)  # [B, H, L, L]

        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        output = eimsum("b h i j, b h j c -> b h i c", attn, v)  # [B, H, L, V]
        output = rearrange(out, "b h l c -> b l (h c)")
        return self._embedding_layer(output)


def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]


def positional_features_exponential(positions, feature_size, seq_length, min_half_life=3.0):
    """
    Create exponentially decaying positional biases

    Args:
        positions: a 1D vector of length 2*N-1 from [-(N-1), -(N-2), ...,N-2, N-1], where N 
        is the length of input sequence.
        feature_size: the number of basis functions
        seq_length: length of input sequence
        min_half_life: smallest half life

    Returns: 
        matrix with dimensions [2*N - 1, feature_size]
    """
    assert seq_length == torch.max(positions) + 1, \
        "seq_length should be max(positions) + 1"
    max_half_life = np.log(seq_len) / np.log(2.0)
    half_life = 2 ** torch.linspace(min_half_life, max_half_life,
                                    feature_size, device=positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    output = torch.exp(-np.log(2.0) / half_life * positions)

    assert (output.shape[:-1] ==
            positions.shape) & (output.shape[-1] == feature_size)
    return torch.exp(-np.log(2.0) / half_life * positions)


def positional_features_central_mask(positions, feature_size, seq_length):
    """
    Create positional feature in which central regions are one and other regions are zero
    """
    assert seq_length == torch.max(torch.abs(positions)) + 1, \
        "seq_length should be max(positions) + 1"

    center_widths = 2 ** torch.arange(1, feature_size + 1,
                                      device=positions.device).float()
    center_widths = center_widths - 1
    output = (center_widths[None, ...] > positions.abs()[..., None]).float()

    assert output.shape[
        :-1] == positions.shape & output.shape[-1] == feature_size
    return output


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) -
                         concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)


def positional_features_gamma(positions, feature_size, seq_length, stddev=None, start_mean=None):
    if stddev is None:
        stddev = seq_length / (2 * feature_size)

    if start_mean is None:
        start_mean = seq_length / features

    mean = torch.linspace(start_mean, seq_len, features,
                          device=positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(positions.float().abs()[..., None],
                              concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities)
    return outputs


def positional_features_all(positions, feature_size, seq_length, symmetric=False):
    """
    Compute relative positional encodings/features.

    Each positional feature function will compute/provide the same fraction of
    features, making up the total of feature_size.

    Args:
    positions: Tensor of relative positions of arbitrary shape.
    feature_size: Total number of basis functions.
    seq_length: Sequence length denoting the characteristic length that
      the individual positional features can use. This is required since the
      parametrization of the input features should be independent of `positions`
      while it could still require to use the total number of features.
    symmetric: If True, the resulting features will be symmetric across the
      relative position of 0 (i.e. only absolute value of positions will
      matter). If false, then both the symmetric and asymmetric version
      (symmetric multiplied by sign(positions)) of the features will be used.

    Returns:
    Tensor of shape: `positions.shape + [feature_size]`.
    """
    assert seq_length == torch.max(positions) + 1, \
        "seq_length should be max(positions) + 1"

    num_components = len(feature_functions)
    if not symmetric:
        num_components = 2 * num_components

    assert feature_size % num_components == 0, (f"feature_size has "
                                                 "to be divisible by {num_components}")

    feature_functions = [positional_features_exponential,
                         positional_features_central_mask,
                         positional_features_gamma]

    num_basis_per_class = feature_size // num_components

    embeddings = [f(torch.abs(positions), num_basis_per_class, seq_length)
                  for f in feature_functions]
    embeddings = torch.cat(embeddings, dim=-1)

    if not symmetric:
        embeddings = torch.cat(embeddings,
                               torch.sign(positions)[..., None] * embeddings,
                               dim=-1)
    return embeddings
