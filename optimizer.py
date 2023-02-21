import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.others import w


class Optimizer(nn.Module):
    def __init__(
        self,
        additional_inp_dim=0,
        preproc=False,
        hidden_sz=20,
        preproc_factor=10.0,
        manual_init_output_params=False,
    ):
        super().__init__()

        self.additional_inp_dim = additional_inp_dim
        self.hidden_sz = hidden_sz

        if preproc:
            self.recurs = nn.LSTMCell(2 + additional_inp_dim, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1 + additional_inp_dim, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)

        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)

        if manual_init_output_params:
            ### init params
            torch.nn.init.xavier_uniform_(self.output.weight)
            torch.nn.init.zeros_(self.output.bias)

    def forward(self, optee_grads, hidden, cell, additional_inp=None):
        assert (
            self.additional_inp_dim == 0 and additional_inp is None
        ) or optee_grads.size()[0] == additional_inp.size()[
            0
        ], "optee_grads and additional_inp should have the same batch size or additional_inp should be None"
        if self.preproc:
            # Implement preproc described in Appendix A

            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            optee_grads = optee_grads.data
            optee_grads_preproc = w(torch.zeros(optee_grads.size()[0], 2))
            keep_grads = (torch.abs(optee_grads) >= self.preproc_threshold).squeeze()
            optee_grads_preproc[:, 0][keep_grads] = (
                torch.log(torch.abs(optee_grads[keep_grads]) + 1e-8)
                / self.preproc_factor
            ).squeeze()
            optee_grads_preproc[:, 1][keep_grads] = torch.sign(
                optee_grads[keep_grads]
            ).squeeze()

            optee_grads_preproc[:, 0][~keep_grads] = -1
            optee_grads_preproc[:, 1][~keep_grads] = (
                float(np.exp(self.preproc_factor)) * optee_grads[~keep_grads]
            ).squeeze()
            optee_grads = w(Variable(optee_grads_preproc))

        if additional_inp is not None:
            optee_grads = torch.cat((optee_grads, additional_inp), dim=-1)
        hidden0, cell0 = self.recurs(optee_grads, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)
