from typing import List, Dict, NamedTuple
from termcolor import colored as clr

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.nn import Module


Constraint = NamedTuple("Constraint", [("mode", Dict[str, Tensor]),
                                       ("elasticity", Dict[str, Tensor])])


class EWC(object):

    def __init__(self, args):
        super(EWC, self).__init__()

        self.merge_elasticities = args.merge_elasticities

        self.scale = args.scale
        self.saved_tasks_no = 0

        self.constraints: List[Constraint] = []

    def __call__(self, model: Module):
        total_elastic_loss = []
        for _idx, constraint in enumerate(self.constraints):
            for name, param in model.named_parameters():
                if param.grad is not None and name in constraint.elasticity:
                    layer_loss = torch.dot(constraint.elasticity[name],
                                           (constraint.mode[name] - param.view(-1)).pow(2))
                    total_elastic_loss.append(layer_loss)

        if total_elastic_loss:
            return self.scale * sum(total_elastic_loss)

        return None

    def end_train_task(self, model, optimizer, train_loader):
        optimizer.zero_grad()

        for _batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = functional.cross_entropy(output, target)
            loss.backward()

        grad = dict({})
        crt_mode = dict({})

        if self.merge_elasticities and self.constraints:
            # Add to previous matrices if in `merge` mode
            elasticity = self.constraints[0].elasticity
        else:
            elasticity = dict({})

        for name, param in model.named_parameters():
            if param.grad is not None:
                crt_mode[name] = param.detach().clone().view(-1)
                grad[name] = param.grad.detach().pow(2).clone().view(-1)
                if name in elasticity:
                    elasticity[name].add_(grad[name]).view(-1)
                else:
                    elasticity[name] = grad[name].clone().view(-1)

        new_constraint = Constraint(crt_mode, elasticity)
        if self.merge_elasticities:
            # Remove old constraints if in `merge` mode
            self.constraints.clear()
        self.constraints.append(new_constraint)

        print(clr(f"There are {len(self.constraints):d} elastic constraints!", attrs=["bold"]))
