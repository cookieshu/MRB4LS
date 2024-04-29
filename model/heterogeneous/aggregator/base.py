import torch.nn as nn
import dgl

from .MyHeteroGraphConv import MyHeteroGraphConv

class HeterogeneousAggregator(nn.Module):
    def __init__(self, graph_embedding_dims, etypes, use_edge_weight=False):
        super().__init__()
        self.use_edge_weight = use_edge_weight
        self.primary_etypes = [x for x in etypes if not x[1].endswith('-by')]
        assert len(graph_embedding_dims) >= 2
        self.layer_dict = nn.ModuleDict()
        for etype in self.primary_etypes:
            layers = nn.ModuleList()
            for i in range(len(graph_embedding_dims) - 2):
                layers.append(
                    MyHeteroGraphConv({
                        e: self.get_layer(graph_embedding_dims[i],
                                          graph_embedding_dims[i + 1], i,
                                          len(graph_embedding_dims) - 1)
                        for e in [etype[1], f'{etype[1]}-by']
                    }, use_edge_weight=self.use_edge_weight))
            layers.append(
                MyHeteroGraphConv({
                    e: self.get_layer(graph_embedding_dims[-2],
                                      graph_embedding_dims[-1],
                                      len(graph_embedding_dims) - 2,
                                      len(graph_embedding_dims) - 1)
                    for e in [etype[1], f'{etype[1]}-by']
                }, use_edge_weight=self.use_edge_weight))
            self.layer_dict[str(etype)] = layers

    def get_layer(self, input_dim, output_dim, current_layer, total_layer):
        raise NotImplementedError

    def single_forward(self, layers, blocks, h):
        for layer, block in zip(layers, blocks):
            h = h+layer(block, h)
        return h

    def forward(self, blocks, input_embeddings):
        different_embeddings = True if isinstance(list(input_embeddings.values())[0], dict) else False
        output_embeddings = {}
        for etype in self.primary_etypes:
            if different_embeddings:
                h = input_embeddings[etype]
            else:
                h = {
                    node_name: input_embeddings[node_name]
                    for node_name in [etype[0], etype[2]]
                }
            assert len(h) == 2
            etype_layers = self.layer_dict[str(etype)]
            if blocks[0].is_block:
                etype_blocks = blocks
            else:
                etype_blocks = [
                    dgl.edge_type_subgraph(block, [etype, (etype[2], f'{etype[1]}-by', etype[0])])
                    for block in blocks
                ]
                for x in etype_blocks:
                    x.is_block = True
            h = self.single_forward(etype_layers, etype_blocks, h)
            assert len(h) == 2, 'Unknown error'
            output_embeddings[etype] = h
        return output_embeddings
