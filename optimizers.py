import tqdm
import torch
from torch import nn
from torch import optim
from models import CMIIModelMM
from layer import N3


class Optimizer(object):
    def __init__(
            self, model: CMIIModelMM, N3: N3, optimizer: optim.Optimizer, scheduler, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.reg = N3
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.scheduler = scheduler

    def epoch(self, e, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'Epoch {e}')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                predictions, factors = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = 0.
                for i in factors:
                    l_reg += self.reg.forward(i)
                l = l_fit + l_reg

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.5f}', lr=f'{self.scheduler.get_last_lr()[0]:.4f}')
            self.scheduler.step()