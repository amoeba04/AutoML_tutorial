import torch
import torch.nn as nn

from DARTS.operations import OPS, ReLUConvBN, FactorizedReduce, Identity
from DARTS.utils import drop_path


class Cell(nn.Module):
    def __init__(self, genotype, c_prev_prev, c_prev, c, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(c_prev_prev, c_prev, c)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(c_prev, c, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(c, op_names, indices, concat, reduction)

    def _compile(self, c, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](c, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkCIFAR(nn.Module):
    def __init__(self, init_channels, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self.layers = layers
        self.auxiliary = auxiliary

        stem_multiplier = 3
        c = init_channels
        c_curr = stem_multiplier * c
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False), nn.BatchNorm2d(c_curr)
        )
        self.cells = nn.ModuleList()

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, c
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, c_prev_prev, c_prev, c_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            c_prev_prev, c_prev = c_prev, cell.multiplier * c_curr
            if i == 2 * layers // 3:
                c_to_auxiliary = c_prev

        if auxiliary:
            pass
            # self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
            # self.global_pooling = nn.AdaptiveAvgPool2d(1)
            # self.classifier = nn.Linear(C_prev, num_classes)
            # TODO: implement auxiliary

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            # if i == 2*self.layers//3:
            #     if self.auxiliary and self.training:
            #         logits_aux = self.auxiliary_head(s1)
            # TODO: implement auxiliary
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
