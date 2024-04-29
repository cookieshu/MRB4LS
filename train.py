import copy
import datetime
import json
import os
import time
import warnings

import dgl
import enlighten
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.early_stop import EarlyStopping
from utils.other import process_metadata, create_model, evaluate, deep_apply, is_single_relation_model, \
    get_dataset_name, is_graph_model, time_since, dict2table

warnings.filterwarnings('ignore')

torch.multiprocessing.set_sharing_strategy('file_system')


def train(args, logger):
    with open(args.metadata_path) as f:
        metadata = json.load(f)
        metadata = process_metadata(metadata, args)
        logger.info(metadata)

    model = create_model(metadata, logger, args).to(args.device)
    logger.info(model)

    task_to_evaluate = [x['name'] for x in metadata['task'] if x['evaluation']]

    model.train()
    best_checkpoint_dict = {
        task: copy.deepcopy(model.state_dict())
        for task in task_to_evaluate
    }
    best_val_metrics_dict = {}

    criterions = {}
    if is_single_relation_model(args):
        assert len(metadata['task']) == 1
    for task in metadata['task']:
        # criterions
        if task['type'] == 'top-k-recommendation':

            original_loss_map = {
                'NGCF': 'bpr',
                'HET-NGCF': 'bpr',
                # TODO
            }
            if args.model_name in original_loss_map and task['loss'] != original_loss_map[args.model_name]:
                logger.warning(
                    'You are using a different type of loss with the type in the paper'
                )

            if task['loss'] == 'binary-cross-entropy':
                criterions[task['name']] = nn.BCEWithLogitsLoss()
            else:
                raise NotImplementedError

        elif task['type'] == 'interaction-attribute-regression':
            raise NotImplementedError
        else:
            raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_full = []
    early_stopping_dict = {
        task['name']: EarlyStopping(args.early_stop_patience)
        for task in metadata['task']
    }

    start_time = time.time()
    writer = SummaryWriter(log_dir=os.path.join(
        args.tensorboard_runs_path,
        f'{args.model_name}-{get_dataset_name(args.dataset_path)}',
        f"{str(datetime.datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')}{'-remark-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}",
    ))

    if args.save_checkpoint:
        for task_name in task_to_evaluate:
            os.makedirs(os.path.join(
                args.checkpoint_path,
                f'{args.model_name}-{get_dataset_name(args.dataset_path)}',
                task_name,
            ),
                exist_ok=True)

    enlighten_manager = enlighten.get_manager()
    batch = 0

    if is_graph_model(args):
        etype2num_neighbors = {
            etype: np.clip(
                np.quantile(model.graph.in_degrees(etype=etype),
                            args.neighbors_sampling_quantile,
                            method='nearest'),  # `interpolation` was renamed to `method`
                args.min_neighbors_sampled, args.max_neighbors_sampled)
            for etype in model.graph.canonical_etypes
        }
        logger.debug(f'Neighbors sampled {etype2num_neighbors}')

    try:
        with enlighten_manager.counter(total=args.num_epochs,
                                       desc='Training epochs',
                                       unit='epochs') as epoch_pbar:
            for epoch in epoch_pbar(range(1, args.num_epochs + 1)):
                if is_graph_model(args):
                    neighbor_sampler = dgl.dataloading.MultiLayerNeighborSampler(
                        [etype2num_neighbors] *
                        (len(args.graph_embedding_dims) - 1))
                else:

                    neighbor_sampler = dgl.dataloading.MultiLayerNeighborSampler([0])

                def edge_sampling(etype):
                    df = pd.DataFrame(
                        torch.stack(model.graph.edges(etype=etype), dim=1).numpy())
                    return df.sample(frac=1).drop_duplicates(0).index.values

                eid_dict = {
                    etype: edge_sampling(etype)
                    for etype in model.primary_etypes
                }

                # parse reverse_etypes
                etypes = copy.deepcopy(model.graph.canonical_etypes)
                reverse_etypes = {}
                for etype in model.graph.canonical_etypes:
                    if etype[1].endswith('-by'):
                        reverse_etype = (etype[2], etype[1][:-len('-by')], etype[0])  # ('user', 'gift', 'room')
                        reverse_etypes[etype] = reverse_etype  # ('room', 'gift-by', 'user'):('user', 'gift', 'room')
                        reverse_etypes[reverse_etype] = etype  # ('user', 'gift', 'room'):('room', 'gift-by', 'user')
                        etypes.remove(etype)
                        etypes.remove(reverse_etype)
                assert len(etypes) == 0

                sampler = dgl.dataloading.as_edge_prediction_sampler(
                    neighbor_sampler, exclude='reverse_types',
                    reverse_etypes=reverse_etypes,
                    negative_sampler=dgl.dataloading.negative_sampler.Uniform(
                        args.negative_sampling_ratio))
                dataloader = dgl.dataloading.DataLoader(
                    model.graph, eid_dict, sampler,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.num_workers
                )

                with enlighten_manager.counter(total=len(dataloader),
                                               desc='Training batches',
                                               unit='batches',
                                               leave=False) as batch_pbar:
                    for input_nodes, positive_graph, negative_graph, blocks in batch_pbar(dataloader):
                        batch += 1
                        if is_graph_model(args):
                            if batch == 1:
                                node_coverage = {
                                    k: len(v) / model.graph.num_nodes(k)
                                    for k, v in input_nodes.items()
                                }
                                logger.debug(
                                    f'Node coverage {deep_apply(node_coverage)}'
                                )
                            input_nodes = {
                                k: v.to(args.device)
                                for k, v in input_nodes.items()
                            }
                        positive_graph = positive_graph.to(args.device)
                        negative_graph = negative_graph.to(args.device)

                        if is_graph_model(args):
                            blocks = [block.to(args.device) for block in blocks]
                            output_embeddings = model.aggregate_embeddings(input_nodes, blocks)
                        else:
                            output_embeddings = None
                        loss = 0
                        for task in metadata['task']:
                            positive_index = torch.stack(
                                positive_graph.edges(etype=task['scheme']))
                            negative_index = torch.stack(
                                negative_graph.edges(etype=task['scheme']))
                            index = torch.cat((positive_index, negative_index),
                                              dim=1)
                            if not is_graph_model(args):
                                # map indexs
                                index[0] = positive_graph.ndata[dgl.NID][
                                    task['scheme'][0]][index[0]]
                                index[1] = positive_graph.ndata[dgl.NID][
                                    task['scheme'][2]][index[1]]
                            first = {
                                'name': task['scheme'][0],
                                'index': index[0]
                            }
                            second = {
                                'name': task['scheme'][2],
                                'index': index[1]
                            }
                            y_pred = model(first, second, task['name'], output_embeddings)
                            if task['loss'] == 'binary-cross-entropy':
                                y_true = torch.cat(
                                    (torch.ones(positive_index.size(1)), torch.zeros(negative_index.size(1)))).to(
                                    args.device)
                                task_loss = criterions[task['name']](y_pred, y_true)
                            else:
                                raise NotImplementedError
                            if task['filename'] == "user-room-gift.csv":
                                weight = args.lamb_1
                            elif task['filename'] == "user-room-chat.csv":
                                weight = args.lamb_2
                            else:
                                weight = 1 - args.lamb_1 - args.lamb_2
                            loss += task_loss * weight

                            if len(metadata['task']) > 1:
                                writer.add_scalar(f"Train/Loss/{task['name']}",
                                                  task_loss.item(), batch)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_full.append(loss.item())
                        writer.add_scalar('Train/Loss', loss.item(), batch)
                        if batch % args.num_batches_show_loss == 0:
                            logger.info(
                                f"Time {time_since(start_time)}, epoch {epoch}, batch {batch}, current loss {loss.item():.4f}, average loss {np.mean(loss_full):.4f}, latest average loss {np.mean(loss_full[-10:]):.4f}"
                            )

                if epoch % args.num_epochs_validate == 0:
                    model.eval()
                    metrics, overall = evaluate(model, [
                        x for x in metadata['task']
                        if x['name'] in task_to_evaluate
                    ], 'val', args)
                    model.train()

                    for task_name, values in metrics.items():
                        for metric, value in values.items():
                            writer.add_scalar(
                                f'Validation/{task_name}/{metric}', value,
                                epoch)

                    logger.info(
                        f"Time {time_since(start_time)}, epoch {epoch}, metrics {deep_apply(metrics)}"
                    )
                    for task_name in copy.deepcopy(task_to_evaluate):

                        early_stop, get_better = early_stopping_dict[
                            task_name](-overall[task_name])
                        if early_stop:
                            task_to_evaluate.remove(task_name)
                            logger.info(f'Task {task_name} early stopped')
                        elif get_better:

                            best_checkpoint_dict[task_name] = copy.deepcopy(
                                model.state_dict())

                            best_val_metrics_dict[task_name] = copy.deepcopy(
                                metrics[task_name])

                            if args.save_checkpoint:
                                torch.save(
                                    {'model_state_dict': model.state_dict()},
                                    os.path.join(
                                        args.checkpoint_path,
                                        f'{args.model_name}-{get_dataset_name(args.dataset_path)}',
                                        task_name, f'ckpt-{epoch}.pt'))

                    if not task_to_evaluate:
                        logger.info('All tasks early stopped')
                        break

    except KeyboardInterrupt:
        logger.info('Stop in advance')


    logger.info(
        f'Best metrics on validation set\n{dict2table(best_val_metrics_dict)}')
    test_metrics_dict = {}

    for task_name, checkpoint in best_checkpoint_dict.items():
        model.load_state_dict(checkpoint)
        model.eval()
        metrics, _ = evaluate(
            model, [x for x in metadata['task'] if x['name'] == task_name], 'test', args)
        test_metrics_dict[task_name] = metrics[task_name]
    logger.info(f'Metrics on test set\n{dict2table(test_metrics_dict)}')
