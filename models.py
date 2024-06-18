import torch
import torch.utils.data
import torch.nn as nn


def get_gpu_memory_usage(tensor):
    memory_usage = tensor.element_size() * tensor.nelement() / (1024 ** 2)
    return memory_usage


def get_model_memory_usage(model):
    memory_usage = 0
    for param in model.parameters():
        memory_usage += param.element_size() * param.nelement()
    return memory_usage  / (1024 ** 2)


def get_model(params):

    if params['model'] == 'ResidualFCNet':
        model = ResidualFCNet(params['input_dim'], params['num_classes'], params['num_filts'], params['depth'], ndevices=torch.cuda.device_count())
        print(f'get_model_memory_usage(model), {get_model_memory_usage(model)} MB')
        
        return model
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

    def __init__(self, num_inputs, num_classes, num_filts, depth=4, ndevices=1):
        super(ResidualFCNet, self).__init__()
        depth = 35
        self.inc_bias = False
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.feat_list = [] # Settings for Model Parallelism
        self.ndevices = ndevices
        
        print(f'self.ndevices: {self.ndevices}')
        layers = []
        layers.append(nn.Linear(num_inputs, num_filts))
        layers.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers.append(ResLayer(num_filts))
        self.feats = torch.nn.Sequential(*layers)
        self.feat_list.append(self.feats)

        layers_2 = []
        layers_2.append(nn.Linear(num_inputs, num_filts))
        layers_2.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers_2.append(ResLayer(num_filts))
        self.feats_2 = torch.nn.Sequential(*layers_2)
        self.feat_list.append(self.feats_2)

        layers_3 = []
        layers_3.append(nn.Linear(num_inputs, num_filts))
        layers_3.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers_3.append(ResLayer(num_filts))
        self.feats_3 = torch.nn.Sequential(*layers_3)
        self.feat_list.append(self.feats_3)

        layers_4 = []
        layers_4.append(nn.Linear(num_inputs, num_filts))
        layers_4.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers_4.append(ResLayer(num_filts))
        self.feats_4 = torch.nn.Sequential(*layers_4)
        self.feat_list.append(self.feats_4)
        
          

    def forward(self, x, class_of_interest=None, return_feats=False):
        
        replicated_x = []
        print('3-1', torch.cuda.memory_allocated() / (1024 ** 2))
        for k, _ in enumerate(self.feat_list):
            d = torch.device("cuda:" + str(k % self.ndevices))
            self.feat_list[k].to(d)
            replicated_x.append(x.to(d))
        print('3-2', torch.cuda.memory_allocated() / (1024 ** 2))
        def get_device(obj):
            if type(obj) is torch.Tensor:
                return obj.device
            return next(obj.parameters()).device
        # print(f'self.feat_list {list(map(get_device, self.feat_list))}')
        # print(f'replicated_x {list(map(get_device, replicated_x))}')

        
        intermidiate_output = []
        for k, _ in enumerate(self.feat_list):
            intermidiate_output.append(self.feat_list[k](replicated_x[k]).to('cuda:0'))
        print('3-3', torch.cuda.memory_allocated() / (1024 ** 2))
        loc_emb = sum(intermidiate_output)
        print(f'get_gpu_memory_usage(intermidiate_output), {get_gpu_memory_usage(intermidiate_output[0])} MB')
        print(f'get_gpu_memory_usage(loc_emb), {get_gpu_memory_usage(loc_emb)} MB')
        print(f'get_gpu_memory_usage(x), {get_gpu_memory_usage(x)} MB')
        print(f'x.shape {x.shape}')
        if return_feats:
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)
        print('3-4', torch.cuda.memory_allocated() / (1024 ** 2))
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
