import torch
import torch.utils.data
import torch.nn as nn

import torch.distributed as dist

def get_model(params):

    if params['model'] == 'ResidualFCNet':
        return ResidualFCNet(params['input_dim'], params['num_classes'], params['num_filts'], params['depth'])
    elif params['model'] == 'LinNet':
        return LinNet(params['input_dim'], params['num_classes'])
    else:
        raise NotImplementedError('Invalid model specified.')

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

class ResidualFCNet(nn.Module):

    def __init__(self, num_inputs, num_classes, num_filts, depth=4):
        super(ResidualFCNet, self).__init__()
        num_filts *= 10
        self.inc_bias = False
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        layers = []
        layers.append(nn.Linear(num_inputs, num_filts))
        layers.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers.append(ResLayer(num_filts))
        layers_2 = []
        layers_2.append(nn.Linear(num_inputs, num_filts))
        layers_2.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers_2.append(ResLayer(num_filts))
        self.feats_1 = torch.nn.Sequential(*layers)
        self.feats_2 = torch.nn.Sequential(*layers_2)

    def forward(self, x, class_of_interest=None, return_feats=False):
        # loc_emb = self.feats_1(x)
        if dist.get_rank() == 0:
            another_gpu = 1
        else:
            another_gpu = 0
        x1 = x.cuda(dist.get_rank())
        x2 = x.cuda(another_gpu)

        self.feats_2 = self.feats_2.cuda(another_gpu)
        loc_emb_1 = self.feats_1(x1)
        loc_emb_2 = self.feats_2(x2)
        loc_emb = loc_emb_1 + loc_emb_2.cuda(dist.get_rank())
        if return_feats:
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)
        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return x @ self.class_emb.weight[class_of_interest, :] + self.class_emb.bias[class_of_interest]
        else:
            return x @ self.class_emb.weight[class_of_interest, :]

class LinNet(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super(LinNet, self).__init__()
        self.num_layers = 0
        self.inc_bias = False
        self.class_emb = nn.Linear(num_inputs, num_classes, bias=self.inc_bias)
        self.feats = nn.Identity()  # does not do anything

    def forward(self, x, class_of_interest=None, return_feats=False):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return x @ self.class_emb.weight[class_of_interest, :] + self.class_emb.bias[class_of_interest]
        else:
            return x @ self.class_emb.weight[class_of_interest, :]
