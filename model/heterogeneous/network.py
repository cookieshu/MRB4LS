import torch
import torch.nn as nn

from .aggregator.RepGAT_Block import RepGAT_Block
from ..general.additive import AdditiveAttention
from ..general.DNNPredictor import  DNNPredictor


class HeterogeneousNetwork(nn.Module):

    def __init__(self, args, graph, tasks):

        super().__init__()
        self.args = args
        self.graph = graph
        self.primary_etypes = [x for x in graph.canonical_etypes if not x[1].endswith('-by')]
        if args.different_embeddings:
            self.embedding = nn.ModuleDict({
                str(etype): nn.ModuleDict({
                    node_name: nn.Embedding(graph.num_nodes(node_name), args.graph_embedding_dims[0])
                    for node_name in [etype[0], etype[2]]
                })
                for etype in self.primary_etypes  # etype('user', 'gift', 'room')
            })
        else:
            self.embedding = nn.ModuleDict({
                node_name: nn.Embedding(graph.num_nodes(node_name), args.graph_embedding_dims[0])
                for node_name in graph.ntypes
            })


        self.aggregator = RepGAT_Block(args.graph_embedding_dims, graph.canonical_etypes, args.num_attention_heads,
                                    use_edge_weight=args.use_edge_weight)
        final_single_embedding_dim = self.args.graph_embedding_dims[-1] * self.args.num_attention_heads

        if args.embedding_aggregator == 'concat':
            embedding_num_dict = {
                node_name: sum([
                    node_name in [etype[0], etype[2]]
                    for etype in self.primary_etypes
                ])
                for node_name in graph.ntypes
            }
            final_embedding_dim_dict = {
                task['name']: (final_single_embedding_dim *
                               embedding_num_dict[task['scheme'][0]],
                               final_single_embedding_dim *
                               embedding_num_dict[task['scheme'][2]])
                for task in tasks
            }
        elif args.embedding_aggregator == 'attn':
            final_embedding_dim_dict = {
                task['name']: (final_single_embedding_dim, final_single_embedding_dim)
                for task in tasks
            }
            self.additive_attention = AdditiveAttention(
                args.attention_query_vector_dim, final_single_embedding_dim)
        else:
            raise NotImplementedError

        self.predictor = nn.ModuleDict({
                task['name']: DNNPredictor(args.dnn_predictor_dims if args.dnn_predictor_dims[0] != -1 else [
                    sum(final_embedding_dim_dict[task['name']]), *args.dnn_predictor_dims[1:]])
                for task in tasks
            })


    def aggregate_embeddings(self, input_nodes, blocks):
        if self.args.different_embeddings:
            input_embeddings = {
                etype: {
                    node_name: self.embedding[str(etype)][node_name](input_nodes[node_name])
                    for node_name in [etype[0], etype[2]]
                }
                for etype in self.primary_etypes
            }
        else:
            input_embeddings = {
                node_name: self.embedding[node_name](input_nodes[node_name])
                for node_name in self.graph.ntypes
            }
        output_embeddings = self.aggregator(blocks, input_embeddings)
        # transpose the nested dict
        output_embeddings = {
            node_name: [
                output_embeddings[etype][node_name]
                for etype in output_embeddings.keys()
                if node_name in output_embeddings[etype]
            ]
            for node_name in self.graph.ntypes
        }
        if self.args.embedding_aggregator == 'concat':

            def embedding_aggregator(x):
                return torch.cat(x, dim=-1)
        elif self.args.embedding_aggregator == 'attn':

            def embedding_aggregator(x):
                return self.additive_attention(torch.stack(x, dim=1))

        output_embeddings = {
            k: embedding_aggregator(v)
            for k, v in output_embeddings.items()
        }
        return output_embeddings

    def forward(self, first, second, task_name, provided_embeddings):
        assert provided_embeddings is not None
        if isinstance(self.predictor, nn.ModuleDict):
            predictor = self.predictor[task_name]
        else:
            predictor = self.predictor
        return predictor(provided_embeddings[first['name']][first['index']],
                         provided_embeddings[second['name']][second['index']])
