from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from sel_conv2d_layer import SelConv2d

class Net(nn.Module):
    def __init__(self, use_attention_improvement):
        super(Net, self).__init__()
        self.use_attention_improvement = use_attention_improvement
        if self.use_attention_improvement:
            self.sconv1 = SelConv2d(1 * 32 * 32, 1, 10, 20, kernel_size=3, padding=1)
            self.sconv2 = SelConv2d(1 * 32 * 32, 10, 20, 40, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1280, 100)
        self.fc2 = nn.Linear(100, 10)

        self.compute_kl_div_loss = False
        self.compute_entropy_loss = False

    def forward(self, x):
        if self.use_attention_improvement:
            _x = x
            x = F.relu(F.max_pool2d(self.sconv1(_x, x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.sconv2(_x, x)), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1280)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def acccumulate_w(self):
        self.sconv1.acccumulate_w()
        self.sconv2.acccumulate_w()

    def reset_accumulated_w(self):
        self.sconv1.reset_accumulated_w()
        self.sconv2.reset_accumulated_w()

    def average_accumulated_w(self):
        self.sconv1.average_accumulated_w()
        self.sconv2.average_accumulated_w()

    def print_accumulated_w(self, print_2nd=True):
        self.sconv1.print_accumulated_w()
        if print_2nd:
            self.sconv2.print_accumulated_w()

    def average_print_reset_accumulated_w(self, print_2nd=True):
        self.average_accumulated_w()
        self.print_accumulated_w(print_2nd)
        self.reset_accumulated_w()

    def use_kl_with_accumulated_w(self, kl_factor, print_w):
        self.compute_kl_div_loss = True
        self.kl_factor = kl_factor
        self.sconv1.use_kl_with_accumulated_w(print_w)
        self.sconv2.use_kl_with_accumulated_w(print_w)

    def use_entropy_loss(self, entropy_factor):
        self.compute_entropy_loss = True
        self.entropy_factor = entropy_factor
        self.sconv1.use_entropy_loss()
        self.sconv2.use_entropy_loss()

    def get_extra_loss(self):
        loss = 0
        if self.compute_kl_div_loss:
            loss -= self.kl_factor * self.sconv1.get_kl_loss()
            loss -= self.kl_factor * self.sconv2.get_kl_loss()

        if self.compute_entropy_loss:
            loss += self.entropy_factor * self.sconv1.get_entropy_loss()
            loss += self.entropy_factor * self.sconv2.get_entropy_loss()

        return loss

    def activate_half_full_depth(self):
        self.sconv1.activate_half_full_depth()
        self.sconv2.activate_half_full_depth()
    def deactivate_half_full_depth(self):
        self.sconv1.deactivate_half_full_depth()
        self.sconv2.deactivate_half_full_depth()

    def freeze_convs(self):
        self.sconv1.freeze_convs()
        self.sconv2.freeze_convs()

    def unfreeze_convs(self):
        self.sconv1.unfreeze_convs()
        self.sconv1.unfreeze_convs()