from typing import Dict, Tuple, List
from features_load import *
import numpy as np
import torch
from models import CMIIModelMM


# DATA_PATH = Path(pkg_resources.resource_filename('kbc', 'data/'))
DATA_PATH = './data'


class Dataset(object):
    def __init__(self, name: str, args):
        self.root = DATA_PATH + '/' + name

        self.data = {}
        self.args = args
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root + '/' + (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis1 = np.max(self.data['train'], axis=0)
        maxis2 = np.max(self.data['valid'], axis=0)
        maxis3 = np.max(self.data['test'], axis=0)
        maxis = [max(maxis1[0], maxis2[0], maxis3[0]), max(maxis1[1], maxis2[1], maxis3[1]), max(maxis1[2], maxis2[2], maxis3[2])]
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(str(self.root + '/' + f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two
        return np.vstack((self.data['train'], copy))

    def eval(
            self, model: CMIIModelMM, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2

            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities

    # def get_filtered_triples(self):
    #     triples_all = []
    #     for f in ['valid', 'test']:  # todo only for test and valid
    #         triples = self.get_examples(f)
    #         print(len(triples))
    #         copy = np.copy(triples)
    #         tmp = np.copy(copy[:, 0])
    #         copy[:, 0] = copy[:, 2]
    #         copy[:, 2] = tmp
    #         copy[:, 1] += self.n_predicates // 2  # has been multiplied by two
    #         triples = np.vstack((triples, copy))
    #         triples_all.append(triples)
    #     triples_all = np.vstack((triples_all[0], triples_all[1]))
    #     return triples_all
    #
    # def mrp(self):
    #     triples = self.get_filtered_triples()
    #     rel_triples = {}
    #     tail_ent = set()
    #     all_ranks, rels, ratio = [], [], []
    #
    #     image_encoder_path, text_encoder_path = clip_model_reload(self.args)
    #     image_encoder = torch.load(image_encoder_path).partial_model
    #     text_encoder = torch.load(text_encoder_path).partial_model
    #
    #     img_features = image_encoder(self.args.img.cuda())
    #     text_feature = text_encoder(self.args.text_features.cuda(), self.args.text_eot_indices.cuda())
    #     img_features = F.normalize(torch.Tensor(img_features), p=2, dim=1)
    #     text_feature = F.normalize(torch.Tensor(text_feature), p=2, dim=1)
    #
    #     ent_vec_img = {i: img_features[i] for i in range(len(img_features))}
    #     ent_vec_text = {i: text_feature[i] for i in range(len(text_feature))}
    #
    #     for triple in triples:  # 按关系分类，统计rel_triples
    #         triple = [i for i in triple]
    #         h, r, t = triple
    #         r_list = rel_triples.get(r, list())
    #         r_list.append(triple)
    #         rel_triples[r] = r_list
    #         tail_ent.add(t)
    #
    #     tails_img_features = img_features[list(tail_ent)]
    #     tails_text_features = text_feature[list(tail_ent)]
    #     for rel, triples in rel_triples.items():
    #         q = torch.from_numpy(triples.astype('int64')).cuda()



























