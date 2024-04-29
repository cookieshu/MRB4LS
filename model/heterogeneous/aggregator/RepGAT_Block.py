import torch.nn as nn

from .base import HeterogeneousAggregator
from .RepGAT import RepGAT


class RepGAT_Block(HeterogeneousAggregator):
    def __init__(self, graph_embedding_dims, etypes, num_attention_heads, use_edge_weight=False):
        self.num_attention_heads = num_attention_heads  # Must put this before super().__init__()
        super().__init__(graph_embedding_dims, etypes, use_edge_weight=use_edge_weight)

    def get_layer(self, input_dim, output_dim, current_layer, total_layer):
        assert 0 <= current_layer <= total_layer - 1
        if current_layer < total_layer - 1:
            # todo 设置dropout
            return RepGAT(
                input_dim *(self.num_attention_heads if current_layer >= 1 else 1),
                output_dim,
                self.num_attention_heads,
                activation=nn.ELU())

        return RepGAT(
            input_dim * (self.num_attention_heads if total_layer >= 2 else 1),
            output_dim, self.num_attention_heads)

    def single_forward(self, layers, blocks, h):
        for layer, block in zip(layers, blocks):
            h = layer(block, h)
            h = {k: v.view(v.size(0), -1) for k, v in h.items()}
        return h



