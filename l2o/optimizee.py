import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from l2o.meta_module import *
from l2o.others import w


### TODO: make a single network class that can be used for all optee nets
class MLPOptee(MetaModule):
    def __init__(
        self,
        act_fn=nn.ReLU(),
        layer_sizes=[20],
        inp_size=28 * 28,
        out_size=10,
        out_act_fn=nn.LogSoftmax(dim=1),
        loss=nn.NLLLoss(),
    ):
        super().__init__()

        ### init layers
        self.layers = {}
        for i, layer_size in enumerate(layer_sizes):
            self.layers[f"mat_{i}"] = MetaLinear(inp_size, layer_size)
            inp_size = layer_size
        self.layers["final_mat"] = MetaLinear(inp_size, out_size)
        self.layers = nn.ModuleDict(self.layers)

        self.act_fn = act_fn
        self.out_act_fn = out_act_fn
        self.loss = loss

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]

    def forward(
        self, data=None, inp=None, out=None, return_acc=False, return_output=False
    ):
        assert (data is None) != (
            inp is None and out is None
        ), "Must provide either data or inp and out"

        if data is not None:
            inp, out = data.sample()
            inp = w(Variable(inp.view(inp.size()[0], 28 * 28)))
            out = w(Variable(out))

        curr_layer = 0
        while f"mat_{curr_layer}" in self.layers:
            inp = self.layers[f"mat_{curr_layer}"](inp)
            inp = self.act_fn(inp)
            curr_layer += 1

        inp = self.layers["final_mat"](inp)
        inp = self.out_act_fn(inp)
        l = self.loss(inp, out)

        if return_acc:
            acc = (inp.argmax(dim=1) == out).float().mean()
            return (l, acc) if not return_output else (l, acc, inp)

        return l if not return_output else (l, inp)


class MNISTNet(MetaModule):
    def __init__(self, layer_size=20, n_layers=1, **kwargs):
        super().__init__()

        inp_size = 28 * 28
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

    def forward(self, data=None, inp=None, out=None, return_acc=False):
        assert (data is None) != (
            inp is None and out is None
        ), "Must provide either data or inp and out"

        if data is not None:
            inp, out = data.sample()
            inp = w(Variable(inp.view(inp.size()[0], 28 * 28)))
            out = w(Variable(out))

        cur_layer = 0
        while f"mat_{cur_layer}" in self.layers:
            inp = self.layers[f"mat_{cur_layer}"](inp)
            inp = self.activation(inp)
            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)

        if return_acc:
            acc = (inp.argmax(dim=1) == out).float().mean()
            return l, acc

        return l


class MNISTNet2Layer(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(n_layers=2, *args, **kwargs)


class MNISTNetBig(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(layer_size=40, *args, **kwargs)


class MNISTNetBig2Layer(MNISTNet):  ### at the edge of GPU memory for unroll=15
    def __init__(self, *args, **kwargs):
        super().__init__(layer_size=80, n_layers=2, *args, **kwargs)


class MNISTSigmoid(MetaModule):
    def __init__(self, layer_sizes=[20], inp_size=28 * 28, **kwargs):
        super().__init__()

        ### init layers
        self.layers = {}
        for i, layer_size in enumerate(layer_sizes):
            self.layers[f"mat_{i}"] = MetaLinear(inp_size, layer_size)
            inp_size = layer_size
        self.layers["final_mat"] = MetaLinear(inp_size, 10)
        self.layers = nn.ModuleDict(self.layers)

        self.activation = nn.Sigmoid()
        self.loss = nn.NLLLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]

    def forward(
        self, data=None, inp=None, out=None, return_acc=False, return_output=False
    ):
        assert (data is None) != (
            inp is None and out is None
        ), "Must provide either data or inp and out"

        if data is not None:
            inp, out = data.sample()
            inp = w(Variable(inp.view(inp.size()[0], 28 * 28)))
            out = w(Variable(out))

        curr_layer = 0
        while f"mat_{curr_layer}" in self.layers:
            inp = self.layers[f"mat_{curr_layer}"](inp)
            inp = self.activation(inp)
            curr_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)

        if return_acc:
            acc = (inp.argmax(dim=1) == out).float().mean()
            return (l, acc) if not return_output else (l, acc, inp)

        return l if not return_output else (l, inp)


class MNISTReLU(MetaModule):
    def __init__(self, layer_sizes=[20], inp_size=28 * 28, **kwargs):
        super().__init__()

        ### init layers
        self.layers = {}
        for i, layer_size in enumerate(layer_sizes):
            self.layers[f"mat_{i}"] = MetaLinear(inp_size, layer_size)
            inp_size = layer_size
        self.layers["final_mat"] = MetaLinear(inp_size, 10)
        self.layers = nn.ModuleDict(self.layers)

        self.activation = nn.ReLU()
        self.loss = nn.NLLLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]

    def forward(
        self, data=None, inp=None, out=None, return_acc=False, return_output=False
    ):
        assert (data is None) != (
            inp is None and out is None
        ), "Must provide either data or inp and out"

        if data is not None:
            inp, out = data.sample()
            inp = w(Variable(inp.view(inp.size()[0], 28 * 28)))
            out = w(Variable(out))

        curr_layer = 0
        while f"mat_{curr_layer}" in self.layers:
            inp = self.layers[f"mat_{curr_layer}"](inp)
            inp = self.activation(inp)
            curr_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)

        if return_acc:
            acc = (inp.argmax(dim=1) == out).float().mean()
            return (l, acc) if not return_output else (l, acc, inp)

        return l if not return_output else (l, inp)


class MNISTRelu(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.ReLU()


class MNISTReluBig(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(layer_size=128, *args, **kwargs)
        self.activation = nn.ReLU()


class MNISTReluBig2Layer(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(layer_size=128, n_layers=2, *args, **kwargs)
        self.activation = nn.ReLU()


class MNISTLeakyRelu(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.LeakyReLU()


class MNISTSoftplus(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.Softplus()


class MNISTReluResidualNormalization(MNISTNet):
    def __init__(self, residual_normalization_end_t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.ReLU()
        self.residual_normalization_end_t = residual_normalization_end_t

    def forward(self, data=None, inp=None, out=None, return_acc=False):
        assert (data is None) != (
            inp is None and out is None
        ), "Must provide either data or inp and out"

        if data is not None:
            inp, out = data.sample()
            inp = w(Variable(inp.view(inp.size()[0], 28 * 28)))
            out = w(Variable(out))

        cur_layer = 0
        interpolation_factor = min(1, t / self.residual_normalization_end_t)
        while f"mat_{cur_layer}" in self.layers:
            inp = self.layers[f"mat_{cur_layer}"](inp)
            if interpolation_factor < 1:
                # apply layer normalization and mix with unnormalized input
                normalized_residual_inp = F.layer_norm(inp, inp.size()[1:])
                inp = (
                    interpolation_factor * inp
                    + (1 - interpolation_factor) * normalized_residual_inp
                )
            inp = self.activation(inp)
            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)

        if return_acc:
            acc = (inp.argmax(dim=1) == out).float().mean()
            return l, acc

        return l


class MNISTSimoidBatchNorm(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.Sigmoid()
        self.batch_norm = MetaBatchNorm1d(num_features=20, **kwargs)

    def forward(self, data=None, inp=None, out=None, return_acc=False):
        assert (data is None) != (
            inp is None and out is None
        ), "Must provide either data or inp and out"

        if data is not None:
            inp, out = data.sample()
            inp = w(Variable(inp.view(inp.size()[0], 28 * 28)))
            out = w(Variable(out))

        cur_layer = 0
        while f"mat_{cur_layer}" in self.layers:
            inp = self.layers[f"mat_{cur_layer}"](inp)
            inp = self.batch_norm(inp)
            inp = self.activation(inp)
            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)

        if return_acc:
            acc = (inp.argmax(dim=1) == out).float().mean()
            return l, acc

        return l


class MNISTReluBatchNorm(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.ReLU()
        self.batch_norm = w(MetaBatchNorm1d(num_features=20, **kwargs))

    def forward(self, data=None, inp=None, out=None, return_acc=False):
        assert (data is None) != (
            inp is None and out is None
        ), "Must provide either data or inp and out"

        if data is not None:
            inp, out = data.sample()
            inp = w(Variable(inp.view(inp.size()[0], 28 * 28)))
            out = w(Variable(out))

        cur_layer = 0
        while f"mat_{cur_layer}" in self.layers:
            inp = self.layers[f"mat_{cur_layer}"](inp)
            inp = self.batch_norm(inp)
            inp = self.activation(inp)
            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)

        if return_acc:
            acc = (inp.argmax(dim=1) == out).float().mean()
            return l, acc

        return l


class MNISTConv(MetaModule):
    """
    From Training Stronger Baselines for Learning to Optimize by T. Chen et. al.
    (https://arxiv.org/pdf/2010.09089.pdf).

    Conv-MNIST: A convolutional neural network (CNN) with 2 convolution layers, 2 max pooling
    layers and 1 fully connected layer on the MNIST dataset. The first convolution layer uses 16 3 x 3
    filters with stride 1. The second convolution layers use 32 5 x 5 filters with stride 1. The max
    pooling layers are of size 2 x 2 with stride 2;
    """

    def __init__(self, n_out=10, act_fn=nn.ReLU, **kwargs):
        super().__init__()

        ### construct layers
        conv_layers = []

        ### block 1
        conv_layers.append(MetaConv2d(1, 16, kernel_size=3, stride=1))
        conv_layers.append(act_fn(inplace=True))
        conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        ### block 2
        conv_layers.append(MetaConv2d(16, 32, kernel_size=5, stride=1))
        conv_layers.append(act_fn(inplace=True))
        conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_layers = MetaSequential(*conv_layers)

        ### fc layer
        self.fc_layer = MetaLinear(4 * 4 * 32, n_out)

        self.loss = nn.NLLLoss()

    @torch.no_grad()
    def reset_parameters(self):
        ### reinit params: random normal distribution w/ standard deviation of 0.01, bias = 0
        for m in self.modules():
            if isinstance(m, MetaLinear) or isinstance(m, MetaConv2d):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        param.data.normal_(0, 0.01)
                    elif "bias" in name:
                        param.data.zero_()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]

    def forward(self, data=None, inp=None, out=None, return_acc=False):
        assert (data is None) != (
            inp is None and out is None
        ), "Must provide either data or inp and out"

        if data is not None:
            inp, out = data.sample()
            inp = w(Variable(inp))
            out = w(Variable(out))

        B = inp.size()[0]
        inp = self.conv_layers(inp)
        inp = inp.view(B, -1)
        inp = self.fc_layer(inp)
        inp = F.log_softmax(inp, dim=1)
        l = self.loss(inp, out)

        if return_acc:
            acc = (inp.argmax(dim=1) == out).float().mean()
            return l, acc

        return l


class MNISTMixtureOfActivations(MetaModule):
    def __init__(
        self,
        layer_size=20,
        n_layers=1,
        activations=("sigmoid", "tanh", "relu"),
        **kwargs,
    ):
        super().__init__()

        inp_size = 28 * 28
        self.layers = {}
        for i in range(n_layers):
            self.layers[f"mat_{i}"] = MetaLinear(inp_size, layer_size)
            inp_size = layer_size

        self.layers["final_mat"] = MetaLinear(inp_size, 10)
        self.layers = nn.ModuleDict(self.layers)

        self.activations = activations
        self.loss = nn.NLLLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]

    def forward(self, data=None, inp=None, out=None, return_acc=False):
        assert (data is None) != (
            inp is None and out is None
        ), "Must provide either data or inp and out"

        if data is not None:
            inp, out = data.sample()
            inp = w(Variable(inp.view(inp.size()[0], 28 * 28)))
            out = w(Variable(out))

        cur_layer = 0
        while f"mat_{cur_layer}" in self.layers:
            inp = self.layers[f"mat_{cur_layer}"](inp)
            ### split along the batch dim into len(self.activations) chunks and apply the activations
            inp = torch.cat(
                [
                    getattr(F, self.activations[i % len(self.activations)])(
                        inp[i :: len(self.activations)]
                    )
                    for i in range(len(self.activations))
                ],
                dim=0,
            )
            ### need to permute the desired outputs as well
            out = torch.cat(
                [out[i :: len(self.activations)] for i in range(len(self.activations))],
                dim=0,
            )

            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)

        if return_acc:
            acc = (inp.argmax(dim=1) == out).float().mean()
            return l, acc

        return l


### CIFAR-10 --------------------------------------------------------------------------------------------
class CIFAR10Sigmoid(MetaModule):
    def __init__(self, layer_size=40, n_layers=2, inp_size=32 * 32 * 3, **kwargs):
        super().__init__()
        self.inp_size = inp_size

        self.layers = {}
        for i in range(n_layers):
            self.layers[f"mat_{i}"] = MetaLinear(inp_size, layer_size)
            inp_size = layer_size

        self.layers["final_mat"] = MetaLinear(inp_size, 10)
        self.layers = ModuleDict(self.layers)

        self.activation = nn.Sigmoid()
        self.loss = nn.NLLLoss()

    @torch.no_grad()
    def reset_parameters(self):
        ### reinit params: random normal distribution w/ standard deviation of 0.01, bias = 0
        for m in self.modules():
            if isinstance(m, MetaLinear):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        param.data.normal_(0, 0.01)
                    elif "bias" in name:
                        param.data.zero_()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]

    def forward(self, data=None, inp=None, out=None, return_acc=False):
        assert (data is None) != (
            inp is None and out is None
        ), "Must provide either data or inp and out"

        if data is not None:
            inp, out = data.sample()
            inp = w(Variable(inp.view(inp.size()[0], self.inp_size)))
            out = w(Variable(out))

        cur_layer = 0
        while f"mat_{cur_layer}" in self.layers:
            inp = self.layers[f"mat_{cur_layer}"](inp)
            inp = self.activation(inp)
            cur_layer += 1

        inp = F.log_softmax(self.layers["final_mat"](inp), dim=1)
        l = self.loss(inp, out)

        if return_acc:
            acc = (inp.argmax(dim=1) == out).float().mean()
            return l, acc

        return l
