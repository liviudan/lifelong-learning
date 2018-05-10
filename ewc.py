from typing import Iterable
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data import DataLoader


Parameters = Iterable[Parameter]
Variables = Iterable[Variable]
Coefficients = Iterable[Iterable[Variable]]


class ElasticConstraint(object):
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 args: Namespace):
        print("Creating an elastic constraint around current parameters.")
        # Copy reference paramters
        self.p_zero = [Variable(p.data.clone()) for p in model.parameters()]

        # Coefficients will always pe positive
        self._coefficients = None
        self._compute_unsigned_coefficients(model, train_loader, args)

        self._coefficients = [[c * c for c in cs] for cs in self._coefficients]

        # Normalize coeefficients
        norm = torch.cat([cff.data.view(-1) for cf_group in self.coefficients
                          for cff in cf_group])\
                    .norm(1)
        for cf_group in self.coefficients:
            for cff in cf_group:
                cff.data.div_(max(norm, 1e-10))

    def _compute_unsigned_coefficients(self, model: nn.Module,
                                       train_loader: DataLoader,
                                       args: Namespace) -> None:
        model.train()  # Model must be in train mode
        model.zero_grad()

        for data, target in train_loader:

            # Perform forward step
            if args.cuda and not data.is_cuda:
                data, target = data.cuda(), target.cuda()
                output = model(Variable(data))

            # Accumulate gradients
            loss = self._loss(output, Variable(target))
            loss.backward()

        print("Constraint has values between: ")
        for param in model.parameters():
            print(param.grad.data.min(), param.grad.data.max())

        self._coefficients = [[Variable(p.grad.data.abs())
                               for p in model.parameters()]]

    def _loss(self, output: Variable, target: Variable) -> Variable:
        return F.cross_entropy(output, target)

    def __call__(self, model: nn.Module) -> Variable:
        losses = []  # type: List[Variable]

        for _t in zip(model.parameters(), self.ref_params, *self.coefficients):
            p_t, p_0, c_t = _t
            diff = p_t - p_0  # type: Variable
            diff = diff * diff
            losses.append(torch.dot(c_t, diff))

        return sum(losses) if losses else None

    @property
    def ref_params(self) -> Variables:
        return self.p_zero

    @property
    def coefficients(self) -> Coefficients:
        return [c for c in self._coefficients]
