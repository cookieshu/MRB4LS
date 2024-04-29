import argparse
from distutils.util import strtobool

import torch


def str2bool(x):
    return bool(strtobool(x))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--non_graph_embedding_dim', type=int, default=64)
    parser.add_argument('--graph_embedding_dims',
                        type=int,
                        nargs='+',
                        default=[64, 64, 64])
    parser.add_argument(
        '--neighbors_sampling_quantile',
        type=float,
        default=0.9,
        help=
        'Set the number of sampled neighbors to the quantile of the numbers of neighbors'
    )
    parser.add_argument(
        '--lamb_1',
        type=float,
        default=0.5,  #
        help='Loss function coefficients for the gift behavior'
    )
    parser.add_argument(
        '--lamb_2',
        type=float,
        default=0.3,
        help='Loss function coefficients for the chat behavior'
    )
    parser.add_argument('--min_neighbors_sampled', type=int, default=4)
    parser.add_argument('--max_neighbors_sampled', type=int, default=512)
    parser.add_argument('--single_attribute_dim', type=int, default=40)
    parser.add_argument('--attention_query_vector_dim', type=int, default=64)
    parser.add_argument(
        '--dnn_predictor_dims',
        type=int,
        nargs='+',
        default=[-1, 128, 1],
        help=
        'You can set first dim as -1 to make it automatically fit the input vector'
    )
    parser.add_argument('--num_batches_show_loss', type=int, default=50)
    parser.add_argument('--num_epochs_validate', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--save_checkpoint', type=str2bool, default=True)
    parser.add_argument('--different_embeddings', type=str2bool, default=False)
    parser.add_argument('--negative_sampling_ratio', type=int, default=4)
    parser.add_argument('--use_edge_weight', type=bool, default=False)
    parser.add_argument('--use_consumption_weight', type=bool, default=False)
    parser.add_argument('--device', type=bool, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    parser.add_argument(
        '--model_name',
        type=str,
        default='MRB4LS',
        choices=[
            'MRB4LS',
        ])
    parser.add_argument('--embedding_aggregator',
                        type=str,
                        default='attn',
                        choices=['concat', 'attn'])
    parser.add_argument('--predictor',
                        type=str,
                        default='dnn',
                        choices=['dot', 'dnn'])
    parser.add_argument('--dataset_path', type=str, default=r'data/ThreeGenres',choices=[r'data/ThreeGenres', r'data/AllGenres'])
    parser.add_argument('--metadata_path', type=str, default='metadata/douyu.json')
    parser.add_argument('--log_path', type=str, default='log')
    parser.add_argument('--tensorboard_runs_path', type=str, default='runs')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint')
    parser.add_argument('--edge_choice',
                        type=int,
                        nargs='+',
                        default=[0,1,2],
                        help='The gift, chat, and enter behaviors are denoted as 0, 1, and 2, respectively.')
    parser.add_argument('--training_task_choice',
                        type=int,
                        nargs='+',
                        default=[],
                        help='Left empty to use all in metadata file')
    parser.add_argument('--evaluation_task_choice',
                        type=int,
                        nargs='+',
                        default=[],
                        help='Left empty to use all in `training_task_choice`')
    parser.add_argument('--task_loss_overwrite', type=str, nargs='+')
    parser.add_argument('--task_weight_overwrite', type=float, nargs='+')
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print(
            'Warning: if you are not in testing mode, you may have got some parameters wrong input'
        )
    return args
