from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from data_loaders import load_datasets, get_dataset_loaders
from network import Net

from ewc import ElasticConstraint

# ------------------- Training settings -------------------
parser = argparse.ArgumentParser(
    description='PyTorch MNIST/CIFAR10 Simple Conv')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--ewc', action='store_true', default=False,
                    help='adds elastic weight consolidation')
parser.add_argument('--elastic-scale', type=float, default=1000,
                    dest='elastic_scale',
                    help='Elastic Scale (default: 10^3)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

load_datasets(args)

# ---------------------------------------------------------


def init_model_and_optimizer(use_attention_improvement=False):
    model = Net(use_attention_improvement)
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return model, optimizer


def test(model, dataset_name):
    train_loader, test_loader = get_dataset_loaders(dataset_name)
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target,
                                     size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        dataset_name, test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy


def train(model, optimizer, epoch, dataset_name, elastic=None):
    train_loader, test_loader = get_dataset_loaders(dataset_name)
    model.train()
    train_loss = 0
    print('\nTraining Epoch: {} on Dataset: {} started.'.format(
        epoch, dataset_name))
    if elastic is not None:
        ce_losses = []
        elastic_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) + model.get_extra_loss()
        if elastic is not None:
            elastic_loss = elastic(model) * args.elastic_scale
            elastic_losses.append(elastic_loss.data[0])
            ce_losses.append(loss.data[0])
            loss = loss + elastic_loss
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx == len(train_loader) - 1:
            train_loss /= len(train_loader.dataset)
            print('Training Epoch: {} on Dataset: {} done.\tAverage loss: {:.6f}\n'.format(
                epoch, dataset_name, train_loss))
            if elastic is not None:
                print("Cross-Entropy vs Elastic loss: ",
                      np.mean(ce_losses), " -- ", np.mean(elastic_losses))


def init_simple_and_attention_run():
    model_simple, optimizer_simple = init_model_and_optimizer()

    def train_simple(epoch, dataset_name):
        return train(model_simple, optimizer_simple, epoch, dataset_name)

    def test_simple(dataset_name):
        return test(model_simple, dataset_name)

    model_attention, optimizer_attention = init_model_and_optimizer(
        use_attention_improvement=True)

    def train_attention(epoch, dataset_name, elastic=None):
        return train(model_attention, optimizer_attention, epoch, dataset_name, elastic=elastic)

    def test_attention(dataset_name):
        return test(model_attention, dataset_name)

    return train_simple, test_simple, model_simple, optimizer_simple, train_attention, test_attention, model_attention, optimizer_attention


print("------------------Starting Sequential traning.------------------")

train_simple, test_simple, model_simple, optimizer_simple, train_attention, test_attention, model_attention, optimizer_attention = init_simple_and_attention_run()

MNIST_results_simple = []
MNIST_results_attention = []
CIFAR10_results_simple = []
CIFAR10_results_attention = []

'''
print("------Starting simple run------")

for i in range(args.epochs):
    train_simple(i + 1 , "MNIST")
    MNIST_results_simple.append(test_simple("MNIST"))
    CIFAR10_results_simple.append(test_simple("CIFAR10"))

for i in range(args.epochs):
    train_simple(i + 1, "CIFAR10")
    MNIST_results_simple.append(test_simple("MNIST"))
    CIFAR10_results_simple.append(test_simple("CIFAR10"))
'''

print("------Starting attention run------")

# model_attention.activate_half_full_depth()

for i in range(args.epochs):
    train_attention(i + 1, "MNIST")
    MNIST_results_attention.append(test_attention("MNIST"))
    CIFAR10_results_attention.append(test_attention("CIFAR10"))

if args.ewc:
    train_loader, _ = get_dataset_loaders("MNIST")
    elastic = ElasticConstraint(model_attention, train_loader, args)
else:
    elastic = None

model_attention.acccumulate_w()
test_attention("MNIST")
model_attention.use_kl_with_accumulated_w(10 ** 3, print_w=False)

# model_attention.deactivate_half_full_depth()

# model_attention.freeze_convs()

for i in range(3):
    train_attention(i + 1, "CIFAR10", elastic=elastic)
    # model_attention.acccumulate_w()
    MNIST_results_attention.append(test_attention("MNIST"))
    # model_attention.average_print_reset_accumulated_w(print_2nd=False)
    CIFAR10_results_attention.append(test_attention("CIFAR10"))


# code for saving some plots
'''

MNIST_results = {
                "Sequential MNIST simple conv": MNIST_results_simple,
                "Sequential MNIST attention conv": MNIST_results_attention,
                }

CIFAR10_results = {
                  "Sequential CIFAR10 simple conv": CIFAR10_results_simple,
                  "Sequential CIFAR10 attention conv": CIFAR10_results_attention,
                  }

df_MNIST = pd.DataFrame(MNIST_results)
df_CIFAR10 = pd.DataFrame(CIFAR10_results)

timestamp = str(int(time.time()))
modifier = "kl"
try:
    os.mkdir("results")
except:
    pass

try:
    os.mkdir("results/" + modifier)
except:
    pass

os.mkdir("results/" + modifier + "/" + timestamp)

get_filename = lambda dataset_name: "results/" + modifier + "/" + timestamp + "/" + dataset_name \
                                    + "_results_sequential_epoch_count_" + str(args.epochs) \
                                    + "_" + modifier

df_MNIST.to_csv(get_filename("MNIST") + ".csv")
df_CIFAR10.to_csv(get_filename("CIFAR10") + ".csv")

df_MNIST.plot()
plt.savefig(get_filename("MNIST") + ".png")
plt.clf()

df_CIFAR10.plot()
plt.savefig(get_filename("CIFAR10") + ".png")
'''
