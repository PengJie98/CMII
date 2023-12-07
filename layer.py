import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.2):
        super(Adapter, self).__init__()
        self.residual_ratio = residual_ratio
        # How about use learnable residual ratio instead ?
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a = self.fc(x)
        x = self.residual_ratio * a + (1 - self.residual_ratio) * x
        return x


class PostmatHead(nn.Module):
    def __init__(self, classifier_head, encoder, rank, bias=False):
        global in_features
        super(PostmatHead, self).__init__()
        if encoder == 'ViT-B/16':
            in_features = 512
        elif encoder == 'RN50':
            in_features = 1024
        elif encoder == 'ViT-L/14':
            in_features = 768
        elif encoder == 'GAT':
            in_features = 200
        elif encoder == 'CompleX':
            in_features = 2 * rank
        else:
            raise ValueError(f"Invalid clip encoder: {encoder}")

        if classifier_head == 'pooling_linear':
            linear_head = nn.Linear(256, 2 * rank, bias=bias)
            nn.init.xavier_uniform_(linear_head.weight.data)
        else:
            linear_head = nn.Linear(in_features, 2 * rank, bias=bias)
            nn.init.xavier_uniform_(linear_head.weight.data)
        self.head = None

        if classifier_head == 'linear':
            self.head = nn.Sequential(
                    linear_head
                )

        elif classifier_head == 'adapter':
            adapter = Adapter(in_features, residual_ratio=0.2)
            self.head = nn.Sequential(
                adapter,
                linear_head,
            )

        elif classifier_head == 'pooling':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(2 * rank)
                # nn.AvgPool1d(4, stride=4)
            )
        elif classifier_head == 'pooling_linear':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(256),
                linear_head
                # nn.AvgPool1d(4, stride=4)
            )
        else:
            raise ValueError(f"Invalid head: {classifier_head}")

    def forward(self, input):
        return self.head(input)


class Attention(nn.Module):
    def __init__(self, clip_encoder):
        global in_features
        super(Attention, self).__init__()
        if clip_encoder == 'ViT-B/16':
            in_features = 512
        elif clip_encoder == 'RN50':
            in_features = 1024
        elif clip_encoder == 'ViT-L/14':
            in_features = 768
        self.text_features = in_features
        self.image_features = in_features
        # self.fc1 = nn.Linear(in_features, 1, bias=False)

    def forward(self, text, images):
        text = text.unsqueeze(1)
        attention_weights = torch.matmul(text, images.permute(0, 2, 1))
        weighted_images = torch.bmm(attention_weights, images).squeeze(-2)
        return weighted_images


class WeightedPred(nn.Module):
    def __init__(self, beta_s, beta_m, beta_t, beta_i):
        super(WeightedPred, self).__init__()

        self.beta_s = nn.Parameter(torch.tensor(beta_s), requires_grad=True)
        self.beta_t = nn.Parameter(torch.tensor(beta_t), requires_grad=True)
        self.beta_i = nn.Parameter(torch.tensor(beta_i), requires_grad=True)
        self.beta_m = nn.Parameter(torch.tensor(beta_m), requires_grad=True)

    def forward(self, entity, mm, text, images):
        total = self.beta_s + self.beta_i + self.beta_t + self.beta_m
        return self.beta_s / total * entity + self.beta_t / total * text + self.beta_i / total * images + self.beta_m / total * mm


class MFB(nn.Module):
    def __init__(self, input_size, output_size, factor_num):
        super(MFB, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.factor_num = factor_num

        self.W1 = nn.Parameter(torch.Tensor(input_size, factor_num))
        self.W2 = nn.Parameter(torch.Tensor(input_size, factor_num))
        self.W3 = nn.Parameter(torch.Tensor(input_size, factor_num))
        self.W4 = nn.Parameter(torch.Tensor(200, factor_num))

        # self.W4 = nn.Parameter(torch.Tensor(input_size, factor_num))
        self.H = nn.Parameter(torch.Tensor(factor_num, output_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3)
        nn.init.xavier_uniform_(self.W4)
        nn.init.xavier_uniform_(self.H)

    def forward(self, x1, x2, x3, x4):
        # x1, x2, x3: (batch_size, input_size)
        x1 = torch.matmul(x1, self.W1)  # x1: (batch_size, factor_num)
        x2 = torch.matmul(x2, self.W2)  # x2: (batch_size, factor_num)
        x3 = torch.matmul(x3, self.W3)  # x3: (batch_size, factor_num)
        x4 = torch.matmul(x4, self.W4)  # x3: (batch_size, factor_num)

        x = torch.relu(x1 * x2 * x3 * x4)  # element-wise multiplication
        x = torch.matmul(x, self.H)  # x: (batch_size, output_size)
        x = F.normalize(x, dim=1)
        return x


class MFB1(nn.Module):
    def __init__(self, input_size, output_size, factor_num):
        super(MFB1, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.factor_num = factor_num

        self.W1 = nn.Parameter(torch.Tensor(768, factor_num))
        # self.W1 = nn.Parameter(torch.Tensor(input_size, factor_num))

        self.W2 = nn.Parameter(torch.Tensor(input_size, factor_num))
        self.W3 = nn.Parameter(torch.Tensor(input_size, factor_num))
        self.W4 = nn.Parameter(torch.Tensor(input_size, factor_num))
        self.H = nn.Parameter(torch.Tensor(factor_num, output_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3)
        nn.init.xavier_uniform_(self.W4)
        nn.init.xavier_uniform_(self.H)

    def forward(self, x1, x2, x3, x4):
        # x1, x2, x3: (batch_size, input_size)

        x1 = torch.matmul(x1, self.W1)  # x1: (batch_size, factor_num)
        x2 = torch.matmul(x2, self.W2)  # x2: (batch_size, factor_num)
        x3 = torch.matmul(x3, self.W3)  # x3: (batch_size, factor_num)
        x4 = torch.matmul(x4, self.W4)  # x3: (batch_size, factor_num)

        x = torch.relu(x1 * x2 * x3 * x4)  # element-wise multiplication
        x = torch.matmul(x, self.H)  # x: (batch_size, output_size)
        x = F.normalize(x, dim=1)
        return x


class N3_abs(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class N3(N3_abs):
    def __init__(self, weight: int = 1e-3):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]


class Attention2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Attention2, self).__init__()

        self.fc1 = nn.Linear(input_dim, 768)
        self.fc = nn.Linear(768, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = torch.prod(x, dim=1)  # learn

        x_image = self.fc1(x)
        x = F.relu(self.fc(x_image))
        return x, x_image


