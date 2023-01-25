import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.others import w
from meta_module import *


class MNISTNet(MetaModule):
    def __init__(self, layer_size=20, n_layers=1, **kwargs):
        super().__init__()

        inp_size = 28*28
        self.layers = {}
        for i in range(n_layers):
            self.layers[f"mat_{i}"] = MetaLinear(inp_size, layer_size)
            inp_size = layer_size

        self.layers["final_mat"] = MetaLinear(inp_size, 10)
        self.layers = nn.ModuleDict(self.layers)

        self.activation = nn.Sigmoid()
        self.loss = nn.NLLLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]
    
    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        out = w(Variable(out))

        cur_layer = 0
        while f"mat_{cur_layer}" in self.layers:
            inp = self.layers[f"mat_{cur_layer}"](inp)
            inp = self.activation(inp)
            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)
        return l


class MNISTNet2Layer(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(n_layers=2, *args, **kwargs)


class MNISTNetBig(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(layer_size=40, *args, **kwargs)

        
class MNISTRelu(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.ReLU()

class MNISTLeakyRelu(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.LeakyReLU()

class MNISTReluResidualNormalization(MNISTNet):
    def __init__(self, residual_normalization_end_t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.ReLU()
        self.residual_normalization_end_t = residual_normalization_end_t

    def forward(self, loss, t):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        out = w(Variable(out))

        cur_layer = 0
        interpolation_factor = min(1, t / self.residual_normalization_end_t)
        while f"mat_{cur_layer}" in self.layers:
            inp = self.layers[f"mat_{cur_layer}"](inp)
            if interpolation_factor < 1:
                # apply layer normalization and mix with unnormalized input
                normalized_residual_inp = F.layer_norm(inp, inp.size()[1:])
                inp = interpolation_factor * inp + (1 - interpolation_factor) * normalized_residual_inp
            inp = self.activation(inp)
            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)
        return l

class MNISTSimoidBatchNorm(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.Sigmoid()
        self.batch_norm = MetaBatchNorm1d(
            num_features=20,
            **kwargs
        )

    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        out = w(Variable(out))

        cur_layer = 0
        while f"mat_{cur_layer}" in self.layers:
            inp = self.layers[f"mat_{cur_layer}"](inp)
            inp = self.batch_norm(inp)
            inp = self.activation(inp)
            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)
        return l

class MNISTReluBatchNorm(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.ReLU()
        self.batch_norm = MetaBatchNorm1d(
            num_features=20,
            **kwargs
        )

    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        out = w(Variable(out))

        cur_layer = 0
        while f"mat_{cur_layer}" in self.layers:
            inp = self.layers[f"mat_{cur_layer}"](inp)
            inp = self.batch_norm(inp)
            inp = self.activation(inp)
            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)
        return l
