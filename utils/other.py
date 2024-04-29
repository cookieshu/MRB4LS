import copy
import datetime
import logging
import math
import os
import time
from itertools import chain

import coloredlogs
import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ndcg_score

from model.heterogeneous import HeterogeneousNetwork
from utils.dir_util import ROOT_DIR
from utils.metrics import recall

# A simple cache mechanism for df reading and sorting, since it will be runs for many times
_df_cache_for_validation = {}


@torch.no_grad()
def evaluate(model, tasks, mode, args):
    metrics = {}
    if is_graph_model(args):
        input_nodes = {node_name: model.graph.nodes(ntype=node_name).to(args.device)
                       for node_name in model.graph.ntypes
                       }
        provided_embeddings = model.aggregate_embeddings(
            input_nodes,
            [model.graph.to(args.device)] * (len(args.graph_embedding_dims) - 1))
    else:
        provided_embeddings = None

    for task in tasks:
        file_path = os.path.join(ROOT_DIR, args.dataset_path, mode, task['filename'])
        if mode == 'val' and file_path in _df_cache_for_validation:
            df = _df_cache_for_validation[file_path]
        else:
            df = pd.read_csv(file_path)
            df.sort_values(df.columns[0], inplace=True)
            if mode == 'val':
                _df_cache_for_validation[file_path] = df
        columns = df.columns.tolist()
        test_data = np.transpose(df.values)
        test_data = torch.from_numpy(test_data).to(args.device)
        first_indexs, second_indexs, y_trues = test_data
        y_preds = []
        y_trues = y_trues.cpu().numpy()

        for i in range(math.ceil(len(df) / (8 * args.batch_size))):
            first_index = first_indexs[i * (8 * args.batch_size):(i + 1) * (8 * args.batch_size)]
            second_index = second_indexs[i * (8 * args.batch_size):(i + 1) * (8 * args.batch_size)]
            first = {'name': columns[0], 'index': first_index}
            second = {'name': columns[1], 'index': second_index}

            y_pred = model(first, second, task['name'], provided_embeddings)
            y_pred = y_pred.cpu().numpy()
            y_preds.append(y_pred)

        y_preds = np.concatenate(y_preds, axis=0)

        if task['type'] == 'top-k-recommendation':
            single_sample_length = df.groupby(columns[0]).size().values
            assert len(
                set(single_sample_length)) == 1, f'The number of {columns[1]}s for different {columns[0]}s should be equal'
            y_trues = y_trues.reshape(-1, single_sample_length[0])
            y_preds = y_preds.reshape(-1, single_sample_length[0])
            metrics[task['name']] = {
                'NDCG@10':
                    ndcg_score(y_trues, y_preds, k=10, ignore_ties=True),
                'NDCG@20':
                    ndcg_score(y_trues, y_preds, k=20, ignore_ties=True),
                'NDCG@40':
                    ndcg_score(y_trues, y_preds, k=40, ignore_ties=True),
                'NDCG@80':
                    ndcg_score(y_trues, y_preds, k=80, ignore_ties=True),
                'Recall@10':
                    recall(y_trues, y_preds, k=10),
                'Recall@20':
                    recall(y_trues, y_preds, k=20),
                'Recall@40':
                    recall(y_trues, y_preds, k=40),
                'Recall@80':
                    recall(y_trues, y_preds, k=80)
            }
        elif task['type'] == 'interaction-attribute-regression':
            raise NotImplementedError
        else:
            raise NotImplementedError
    overall = {k: np.mean(list(v.values())) for k, v in metrics.items()}
    return metrics, overall


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory) if 'keep' not in x
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def create_model(metadata, logger, args):
    num_nodes_dict = {
        node['name']: len(pd.read_csv(os.path.join(args.dataset_path, 'train', node['filename'])))
        for node in metadata['graph']['node']
    }
    for node in metadata['graph']['node']:
        if len(node['attribute']) != 0:
            logger.warning(
                f"The attributes of node {node['name']} are ignored")

    if is_single_relation_model(args):
        assert len(metadata['graph']['node']) == 2
        assert len(metadata['graph']['edge']) == 1
        # first_node_name=user，second_node_name=item
        first_node_name = metadata['graph']['edge'][0]['scheme'][0]
        second_node_name = metadata['graph']['edge'][0]['scheme'][2]
    else:

        assert len(metadata['graph']['edge']) > 1
    graph_data = {}
    weight_list = {}
    for edge in metadata['graph']['edge']:
        # edge['scheme'] = ('user', 'gift', 'room')
        if edge['scheme'][0] == edge['scheme'][2]:
            raise NotImplementedError
        df = pd.read_csv(os.path.join(args.dataset_path, 'train', edge['filename']))
        if args.use_edge_weight and edge['weighted']:
            if edge['scheme'][1] == 'gift' and args.use_consumption_weight:
                weight_list[edge['scheme']] = torch.as_tensor(df.iloc[:, 3].values)
            else:
                weight_list[edge['scheme']] = torch.as_tensor(df.iloc[:, 2].values)
        graph_data[edge['scheme']] = (torch.as_tensor(df.iloc[:, 0].values), torch.as_tensor(df.iloc[:, 1].values))

    graph = dgl.heterograph(add_reverse(graph_data), num_nodes_dict)
    if is_graph_model(args):
        for edge in metadata['graph']['edge']:
            if args.use_edge_weight and edge['weighted']:
                graph.edges[edge['scheme']].data['weight'] = weight_list[edge['scheme']]
                reverse_edge = (edge['scheme'][2], edge['scheme'][1] + '-by', edge['scheme'][0])
                graph.edges[reverse_edge].data['weight'] = weight_list[edge['scheme']]
        model = HeterogeneousNetwork(args, graph, metadata['task'])
        return model
    raise NotImplementedError(
        f'This model {args.model_name} is under development')


# TODO create_logger
def create_logger(args):
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG',
                        logger=logger,
                        fmt='%(asctime)s %(levelname)s %(message)s')
    log_dir = os.path.join(
        args.log_path,
        f'{args.model_name}-{get_dataset_name(args.dataset_path)}')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(
        log_dir,
        f"{str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')}{'-remark-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}.txt"
    )
    logger.info(f'Check {log_file_path} for the log of this run')
    file_hander = logging.FileHandler(log_file_path)
    logger.addHandler(file_hander)
    return logger


def remove_log_file(logger):
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]


def is_graph_model(args):
    if args.model_name in ['MRB4LS', ]:
        return True

    raise NotImplementedError


def is_single_relation_model(args):
    if args.model_name in ['GCN', 'LightGCN', 'GAT', 'GATv2']:
        return True
    if args.model_name in ['MRB4LS']:
        return False

    raise NotImplementedError


def process_metadata(metadata, args):

    def parse_scheme_from_filename(filename):
        filename = filename.split('.')[0].split('-')
        assert len(filename) == 3
        # (user,gift,room)
        return tuple(filename[x] for x in [0, 2, 1])

    for node in metadata['graph']['node']:
        node['name'] = os.path.splitext(node['filename'])[0]

    for edge in metadata['graph']['edge']:
        edge['scheme'] = parse_scheme_from_filename(edge['filename'])

    for task in metadata['task']:
        task['scheme'] = parse_scheme_from_filename(task['filename'])
        task['name'] = os.path.splitext(task['filename'])[0]

    if args.edge_choice:
        metadata['graph']['edge'] = [
            metadata['graph']['edge'][x] for x in args.edge_choice
        ]

    node_from_edge = set(
        chain.from_iterable([[edge['scheme'][0], edge['scheme'][2]]
                             for edge in metadata['graph']['edge']]))

    metadata['graph']['node'] = [
        x for x in metadata['graph']['node'] if x['name'] in node_from_edge
    ]


    training_task_choice = args.training_task_choice if args.training_task_choice else list(
        range(len(metadata['task'])))
    evaluation_task_choice = args.evaluation_task_choice if args.evaluation_task_choice else training_task_choice
    assert set(evaluation_task_choice) <= set(
        training_task_choice
    ), 'There are tasks in evaluation but not in training'
    metadata['task'] = [{
        **metadata['task'][x], 'evaluation': (x in evaluation_task_choice)
    } for x in training_task_choice]

    if args.task_loss_overwrite is not None:
        assert len(metadata['task']) == len(args.task_loss_overwrite)
        for task, loss in zip(metadata['task'], args.task_loss_overwrite):
            task['loss'] = loss
    if args.task_weight_overwrite is not None:
        assert len(metadata['task']) == len(args.task_weight_overwrite)
        for task, weight in zip(metadata['task'], args.task_weight_overwrite):
            task['weight'] = weight
    assert any([x['weight'] > 0 for x in metadata['task']
                ]), 'Make sure at least one task with positive weight'
    assert set([task['filename'] for task in metadata['task']]) <= set([
        edge['filename'] for edge in metadata['graph']['edge']
    ]), 'There are files in task metadata but not in graph edge metadata'

    return metadata


def add_reverse(graph_data):
    '''
    Add reverse edges for graph data before feed into `dgl.heterograph`
    '''
    for scheme in list(graph_data.keys()):
        if scheme[0] == scheme[2]:
            graph_data[scheme] = (
                torch.cat(graph_data[scheme]),
                torch.cat(graph_data[scheme][::-1]),
            )
        else:
            reversed_scheme = (scheme[2], f'{scheme[1]}-by', scheme[0])  # ('room', 'gift-by', 'user')
            graph_data[reversed_scheme] = (graph_data[scheme][1],
                                           graph_data[scheme][0])
    return graph_data


def copy_arguments(f):
    def selectively_copy(x):
        if isinstance(x, list) or isinstance(x, dict):
            return copy.deepcopy(x)
        else:
            return x
    def wrapper(*args, **kwargs):
        args = tuple(selectively_copy(x) for x in args)
        kwargs = {k: selectively_copy(v) for k, v in kwargs.items()}

        return f(*args, **kwargs)

    return wrapper


@copy_arguments
def deep_apply(d, f=lambda x: f'{x:.4f}'):

    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = deep_apply(v, f)
        else:
            d[k] = f(v)
    return d


def dict2table(d, k_fn=str, v_fn=lambda x: f'{x:.4f}'):
    '''
    Convert a nested dict to markdown table
    '''

    def parse_header(d, depth=0):
        if isinstance(list(d.values())[0], dict):
            header = parse_header(list(d.values())[0], depth=depth + 1)
            for v in d.values():
                assert header == parse_header(v, depth=depth + 1)
            return header
        else:
            return f"| {' | '.join([''] * depth + list(map(k_fn, d.keys())))} |"

    def parse_content(d, accumulated_keys=[]):
        if isinstance(list(d.values())[0], dict):
            contents = []
            for k, v in d.items():
                contents.extend(parse_content(v, accumulated_keys + [k_fn(k)]))
            return contents
        else:
            return [
                f"| {' | '.join(accumulated_keys + list(map(v_fn, d.values())))} |"
            ]

    lines = [parse_header(d), *parse_content(d)]
    return '\n'.join(lines)


def get_dataset_name(dataset_path):
    return dataset_path.split('/')[-1]


def get_abspath(relative_path):
    return os.path.abspath(relative_path)
