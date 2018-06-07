from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SelLinear(nn.Module):

    def __init__(self, in_depth, out_depth, full_depth, **kwargs):
        super(SelLinear, self).__init__()
        self.in_depth = in_depth
        self.out_depth = out_depth
        self.full_depth = full_depth
        self.attention = nn.Linear(1, out_depth * full_depth)
        self.linear = nn.Linear(in_depth, full_depth, **kwargs)
        self.acccumulate_w_activated = False
        self.compute_kl_divergence_loss = False
        self.accumulted_count = 0
        self.use_half_of_full_depth = False
        self.compute_entropy_loss = False

    def forward(self, z, task):
        batch_size = z.size(0)
        task = Variable(torch.Tensor([task]).repeat(batch_size).view(batch_size, 1).cuda())
        out_depth, full_depth = self.out_depth, self.full_depth
        y = self.linear(z)  # y is batch_size x full_depth
        map_size = y.size()[-2:]
        w = self.attention(task)
        # print(x.view(batch_size, -1).size())
        w = w.view(batch_size, out_depth, full_depth)
        # w is batch_size x out_depth x full_depth
        w = F.softmax(w * 100, dim = -1)

        if self.use_half_of_full_depth:
            w = w[:, :, :full_depth // 2]
            q = Variable(torch.zeros(w.size())).cuda()
            w = torch.cat([w, q], dim = 2)

        if self.acccumulate_w_activated:
            w_sum = w.sum(dim=0).data.clone()
            try:
                self.accumulated_w += w_sum
            except AttributeError:
                self.accumulated_w = w_sum
            self.accumulted_count += batch_size

        if self.compute_kl_divergence_loss:
            w_sum_avg = w.sum(dim=0) / batch_size
            kl_divergence = F.kl_div(w_sum_avg, Variable(self.accumulated_w))
            self.kl_loss = kl_divergence

        if self.compute_entropy_loss:
            w_clamped = w.clamp(min = 0.00001)
            entropy = w * torch.log(w_clamped)
            self.entropy_loss = entropy.sum()

        z = torch.bmm(w, y.view(batch_size, full_depth, -1))
        return z.view(batch_size, out_depth)

    def acccumulate_w(self):
        self.acccumulate_w_activated = True
        
    def reset_accumulated_w(self):
        #del self.accumulated_w
        self.accumulted_count = 0
        self.acccumulate_w_activated = False

    def average_accumulated_w(self):
        self.accumulated_w /= self.accumulted_count

    def print_accumulated_w(self):
        print(self.accumulated_w)

    def use_kl_with_accumulated_w(self, print_w=False):
        self.average_accumulated_w()
        self.kl_accumulated_w = self.accumulated_w
        #print(self.accumulated_w)
        #self.kl_accumulated_w.requires_grad = False
        self.reset_accumulated_w()
        self.compute_kl_divergence_loss = True

        if print_w:
            print(self.kl_accumulated_w)

    def get_kl_loss(self):
        if self.compute_kl_divergence_loss:
            return self.kl_loss

    def activate_half_full_depth(self):
        self.use_half_of_full_depth = True
    def deactivate_half_full_depth(self):
        self.use_half_of_full_depth = False

    def use_entropy_loss(self):
        self.compute_entropy_loss = True

    def get_entropy_loss(self):
        if self.compute_entropy_loss:
            return self.entropy_loss

    def _change_convs_freeze_state(self, frozen):
        if frozen:
            req_grad = False
        else:
            req_grad = True

        for param in self.conv.parameters():
            param.requires_grad = req_grad

    def freeze_convs(self):
        self._change_convs_freeze_state(frozen=True)

    def unfreeze_convs(self):
        self._change_convs_freeze_state(frozen=False)