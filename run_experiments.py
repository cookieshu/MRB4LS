from parameters import parse_args
from train import train
from utils.other import create_logger, get_dataset_name, remove_log_file
import warnings

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    args = parse_args()
    models = ['MRB4LS', ]
    neighbors_sampling_quantile = 0.9
    embedding_aggregator = ['concat', 'attn']
    edge_choice = [0, 1, 2]
    dataset_path = [r'data/ThreeGenres', r'data/AllGenres']
    metadata_path = [r'metadata/douyu.json']
    # ==================================================================================================================
    '''设置固定参数'''
    args.num_epochs = 100
    args.batch_size = 1024
    args.embedding_aggregator = 'attn'
    # 数据
    args.dataset_path, args.metadata_path = r'data/ThreeGenres', r'metadata/douyu.json'
    args.use_edge_weight = True
    args.edge_choice = [0, 1, 2]
    args.model_name = 'MRB4LS'
    args.num_attention_heads = 8

    logger = create_logger(args)

    logger.info(f"lamb_1={args.lamb_1}，lamb_2={args.lamb_2}")
    logger.info(args)
    logger.info(
        f'Training model {args.model_name} with dataset {get_dataset_name(args.dataset_path)}'
    )
    train(args, logger)
    remove_log_file(logger)
