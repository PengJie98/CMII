import pickle
import numpy as np
import os
import h5py
import torch


def get_backbone_name(clip_encoder):
    return clip_encoder.replace("/", "-")


def get_image_encoder_name(clip_encoder, image_layer_idx):
    return "_".join([get_backbone_name(clip_encoder), str(image_layer_idx)])


def get_text_encoder_name(clip_encoder, text_layer_idx):
    return "_".join([get_backbone_name(clip_encoder), str(text_layer_idx)])


def get_view_name(image_augmentation, image_views=1):
    name = f"{image_augmentation}"
    if image_augmentation != "none":
        assert image_views > 0
        name += f"_view_{image_views}"
    return name


def get_image_encoder_dir(feature_dir, clip_encoder, image_layer_idx):
    image_encoder_path = os.path.join(
        feature_dir,
        'image',
        get_image_encoder_name(clip_encoder, image_layer_idx)
    )
    return image_encoder_path


def get_image_features_path(dataset,
                            feature_dir,
                            clip_encoder,
                            image_layer_idx):
    test_features_path = os.path.join(
        get_image_encoder_dir(feature_dir, clip_encoder, image_layer_idx),
        dataset,
        "image.h5"
    )
    return test_features_path


def get_text_encoder_dir(feature_dir,
                         clip_encoder,
                         text_layer_idx):
    text_encoder_path = os.path.join(
        feature_dir,
        'text',
        get_text_encoder_name(clip_encoder, text_layer_idx)
    )
    return text_encoder_path


def get_text_features_path(dataset,
                           feature_dir,
                           clip_encoder,
                           text_layer_idx,
                           text_augmentation):
    text_features_path = os.path.join(
        get_text_encoder_dir(feature_dir, clip_encoder, text_layer_idx),
        dataset,
        f"{text_augmentation}.h5")
    return text_features_path


def clip_model_reload(args):
    image_encoder_dir = get_image_encoder_dir(
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")

    text_encoder_dir = get_text_encoder_dir(
        args.feature_dir,
        args.clip_encoder,
        args.text_layer_idx
    )
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
    return image_encoder_path, text_encoder_path


def pretrained_features_load(args):
    text_features_path = get_text_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.text_layer_idx,
        args.text_augmentation
    )
    text_features_f = h5py.File(text_features_path, 'r')
    # text_features['features'] = torch.nn.functional.normalize(text_features['features'], dim=1)

    img_features_path = get_image_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    img_features_f = h5py.File(img_features_path, 'r')
    return text_features_f, img_features_f


def read_entity_from_id(path):

    entity2id = {}
    with open(path + 'entity2id.txt', 'r') as f:
        for line in f:
            instance = line.strip().split()
            entity2id[instance[0]] = int(instance[1])

    return entity2id


def read_relation_from_id(path):
    relation2id = {}
    with open(path + 'relation2id_reverse.txt', 'r') as f:
        for line in f:
            instance = line.strip().split()
            relation2id[instance[0]] = int(instance[1])

    return relation2id


# Load data triples and adjacency matrix
def load_data(args):
    path = '{}/{}/Text/'.format(args.root, args.dataset)

    text_features_f, img_features_f = pretrained_features_load(args)
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)

    if args.image_views == 1:
        img_features = img_features_f['features'][:][::10][:len(entity2id.keys())]
    else:
        out = np.arange(0, args.image_views)
        out = np.tile(out, len(entity2id.keys()))
        index = np.array([i * 10 for i in range(len(entity2id.keys()))]).repeat(args.image_views) + out
        img_features = img_features_f['features'][:][index]

    entity_features = text_features_f['entity_features'][:]
    entity_eot_indices = text_features_f['entity_eot_indices'][:]
    relation_feature = text_features_f['relation_features'][:]
    relation_eot_indices = text_features_f['relation_eot_indices'][:]
    text_features = text_features_f['deps_features'][:]
    text_eot_indices = text_features_f['deps_eot_indices'][:]

    deps_str = text_features_f['deps_str_prompts'][:]
    entity_str = text_features_f['entity_str_prompts'][:]
    relation_str = text_features_f['relation_str_prompts'][:]

    struct_entity = pickle.load(open(args.root + "/features/structure/{}/GAT_entity.pkl".format(args.dataset), 'rb'))
    struct_relation = pickle.load(open(args.root + "/features/structure/{}/GAT_relation.pkl".format(args.dataset), 'rb'))

    return entity2id, relation2id, img_features, (text_features, text_eot_indices, deps_str), (entity_features, entity_eot_indices, entity_str), \
           (relation_feature, relation_eot_indices, relation_str), (struct_entity, struct_relation)