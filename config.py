import argparse

parser = argparse.ArgumentParser()

big_datasets = ['FB15K', 'WN18', 'FB15K-237', 'YAGO15K']
datasets = big_datasets

parser.add_argument(
    '--dataset', choices=datasets, default='WN18',
    help="Dataset in {}".format(datasets))

models = ['MGClipComplExV3']

parser.add_argument(
    '--model', choices=models, default='MGClipComplExV3',
    help="Model in {}".format(models))

optimizers = ['Adagrad', 'Adam', 'SGD', 'RMSProp']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',  # Adagrad for FB15K and YAGO15K
    help="Optimizer in {}".format(optimizers))

parser.add_argument(
    '--max_epochs', default=35, type=int,  # 50
    help="Number of epochs.")

parser.add_argument(
    '--valid', default=1, type=float,
    help="Number of epochs before valid.")

parser.add_argument(
    '--rank', default=500, type=int,
    help="Factorization rank.")

parser.add_argument(
    '--batch_size', default=1024, type=int,
    help="Factorization rank.")

parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate")

parser.add_argument(
    '--trainable_beta', default=True, type=bool,
    help="decay rate for second moment estimate in Adam")

parser.add_argument(
    '--beta_s', default=0.25, type=float,  # 0.3
    help="decay rate for second moment estimate in Adam")

parser.add_argument(
    '--beta_m', default=0., type=float,  # 0.3
    help="decay rate for second moment estimate in Adam")

parser.add_argument(
    '--beta_i', default=0.25, type=float,  # 0.2
    help="decay rate for second moment estimate in Adam")

parser.add_argument(
    '--beta_t', default=0.25, type=float,  # 0.2
    help="decay rate for second moment estimate in Adam")

parser.add_argument(
    '--beta_st', default=0.25, type=float,  # 0.2
    help="decay rate for second moment estimate in Adam")

parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale")

parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam")

parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam")

parser.add_argument(
    '--regularizer', default='N3')

parser.add_argument(
    "--text_layer_idx", type=int, default=0, nargs="?",
    help="means partial finetuning layers")

parser.add_argument(
    "--image_layer_idx", type=int, default=0, nargs="?",
    help="means partial finetuning layers")

parser.add_argument(
    "--root", type=str, default="/home/dw/backup_pj/data", nargs="?",
    help="the root path of features and datasets")

parser.add_argument(
    "--image_views", type=int, default=5, nargs="?",
    help="how many imgs to represents an entity")

parser.add_argument(
    "--feature_dir", type=str, default='/home/dw/backup_pj/data/features', nargs="?",
    help="the pretrained features reload")

parser.add_argument(
    "--clip_encoder", type=str, default='ViT-L/14', nargs="?",
    help="which clip model to load")

parser.add_argument(
    "--text_augmentation", type=str, default='classname', nargs="?",
    help="the text augmentations")

parser.add_argument(
    "--image_features", type=int, default=1, nargs="?",
    help="whether to use image features")

parser.add_argument(
    "--struct_features", type=int, default=1, nargs="?",
    help="whether to use structure features")

parser.add_argument(
    "--text_features", type=int, default=1, nargs="?",
    help="whether to use text features")

parser.add_argument(
    "--entity_features", type=int, default=1, nargs="?",
    help="whether to use entity features")

parser.add_argument(
    "--relation_features", type=int, default=1, nargs="?",
    help="whether to use relation features")

parser.add_argument(
    "--classifier_head", type=str, default='partial tuning', nargs="?",
    help="clip classifier heda", choices=['linear', 'partial tuning', 'adapter'])
