# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer."""

import math

import torch
import torch.nn.init as init

from .initialize import get_model_parallel_world_size
from .layers import ColumnParallelLinear
from .layers import RowParallelLinear

from .utils import divide
from .utils import split_tensor_along_last_dim


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GPT2ParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer for GPT2.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method=unscaled_init_method(0.02), output_layer_init_method=None):
        super(GPT2ParallelSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear(hidden_size, 3*hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)


    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ltor_mask):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, hp]
        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # Raw attention scores. [b, np, s, s]
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.hidden_size_per_attention_head)
        # Apply the left to right attention mask.
        attention_scores = torch.mul(attention_scores, ltor_mask) - \
                           10000.0 * (1.0 - ltor_mask)

        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output

    def __repr__(self):
        return 'GPT2ParallelSelfAttention({}, {}, {}, {})'.format(
            self.hidden_size, self.num_attention_heads,
            self.attention_dropout_prob, self.output_dropout_prob
        )

@torch.jit.script
def gelu_impl(x):
     """OpenAI's gelu implementation."""
     return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                        (1.0 + 0.044715 * x * x)))

def gelu(x):
    return gelu_impl(x)


class GPT2ParallelMLP(torch.nn.Module):
    """MLP for GPT2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(self, hidden_size, output_dropout_prob,
                 init_method=unscaled_init_method(0.02),
                 output_layer_init_method=None):
        super(GPT2ParallelMLP, self).__init__()
        self.hidden_size = hidden_size
        self.output_dropout_prob = output_dropout_prob
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4*hidden_size,
                                                  gather_output=False,
                                                  init_method=init_method)
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4*hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method)
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output

    def __repr__(self):
        return 'GPT2ParallelMLP({}, {})'.format(
            self.hidden_size, self.output_dropout_prob
        )