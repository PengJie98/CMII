# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict
from features_load import load_data
import torch
import torch.nn.functional as F
from torch import optim
from datasets import Dataset
from models import MGClipComplExV3
from optimizers import Optimizer
from config import parser
from torch.optim.lr_scheduler import StepLR, ExponentialLR

args = parser.parse_args()
entity2id, relation2id, img_features, (text_features, text_eot_indices, deps_str), \
    (entity_features, entity_eot_indices, entity_str), (relation_feature, relation_eot_indices, relation_str), (struct_entity, struct_relation) = load_data(args)

img = F.normalize(torch.Tensor(img_features), p=2, dim=1)
entity = F.normalize(torch.Tensor(entity_features), p=2, dim=1)
text = F.normalize(torch.Tensor(text_features), p=2, dim=1) 

if args.image_features:
    args.img = torch.Tensor(img_features)
if args.text_features:
    args.text_features = torch.Tensor(text_features)
    args.text_eot_indices = torch.Tensor(text_eot_indices)
if args.entity_features:
    args.entity_features = torch.Tensor(entity_features)
    args.entity_eot_indices = torch.Tensor(entity_eot_indices)
if args.relation_features:
    args.relation_features = torch.Tensor(relation_feature)
    args.relation_eot_indices = torch.Tensor(relation_eot_indices)
if args.struct_features:
    args.struct_entity = struct_entity
    args.struct_relation = struct_relation

args.entity2id = entity2id
args.relation2id = relation2id

dataset = Dataset(args.dataset, args)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

model = {'MGClipComplExV2': lambda: MGClipComplExV3(args, dataset.get_shape(), args.init),}[args.model]()
print(model)

device = 'cuda'
model.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.1),
    'RMSProp': lambda: optim.RMSprop(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

scheduler = StepLR(optim_method, gamma=0.1, step_size=75)
optimizer = Optimizer(model, optimizer=optim_method, scheduler=scheduler, batch_size=args.batch_size)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}


cur_loss = 0
curve = {'train': [], 'valid': [], 'test': []}
hit10_best = 0.
for e in range(args.max_epochs):
    cur_loss = optimizer.epoch(e, examples)

    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in ['valid', 'test', 'train']
        ]

        if valid['hits@[1,3,10]'][2] > hit10_best:
            hit10_best = valid['hits@[1,3,10]'][2]
            torch.save(model.state_dict(), './data/{}/MGClip/best_model.pth'.format(args.dataset))

        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("\t TRAIN: ", train)
        print("\t TEST: ", test)
        print("\t VALID : ", valid)