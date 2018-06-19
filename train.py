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
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--ewc', action='store_true', default=False,
                    help='adds elastic weight consolidation')
parser.add_argument('--elastic-scale', type=float, default=10 ** 6,
                    dest='elastic_scale',
                    help='Elastic Scale (default: 10^3)')
parser.add_argument('--baseline', action='store_true', default=False,
                    help='runs the baseline instead')
parser.add_argument('--combo_training', action='store_true', default=False,
                    help='runs simultaneous learning instead')
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
        l = F.cross_entropy(output, target)
        #print(l)
        loss = l + model.get_extra_loss()
        if elastic is not None:
            elastic_loss = elastic(model) * args.elastic_scale
            elastic_losses.append(elastic_loss.data[0])
            ce_losses.append(loss.data[0])
            loss = loss + elastic_loss
        loss.backward(retain_graph=True)
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

    def train_simple(epoch, dataset_name, elastic=None):
        return train(model_simple, optimizer_simple, epoch, dataset_name, elastic=elastic)

    def test_simple(dataset_name):
        return test(model_simple, dataset_name)

    model_attention, optimizer_attention = init_model_and_optimizer(
        use_attention_improvement=True)

    def train_attention(epoch, dataset_name, elastic=None):
        return train(model_attention, optimizer_attention, epoch, dataset_name, elastic=elastic)

    def test_attention(dataset_name):
        return test(model_attention, dataset_name)

    return train_simple, test_simple, model_simple, optimizer_simple, train_attention, test_attention, model_attention, optimizer_attention


train_simple, test_simple, model_simple, optimizer_simple, train_attention, test_attention, model_attention, optimizer_attention = init_simple_and_attention_run()

MNIST_results = []
CIFAR10_results = []
SVHN_results = []
FashionMNIST_results = []

tasks = {0: "MNIST",
         1: "CIFAR10",
         2: "SVHN",
         3: "FashionMNIST",
         }

task_results = {0: MNIST_results,
                1: CIFAR10_results,
                2: SVHN_results,
                3: FashionMNIST_results,
                }

if not args.combo_training:
    print("------------------Starting Sequential traning.------------------")
else:
    print("------------------Starting Simultaneous traning.------------------")
    tasks[4] = "COMBO"
    COMBO_results = []

    task_results[4] = COMBO_results

if args.baseline:
    print("------Starting simple run------")

    def train_task(task_id, epochs):
        print("\n---Training on task " + tasks[task_id] + "---")

        for i in range(epochs):
            train_simple(i + 1, tasks[task_id])

            for task in tasks:
                result = test_simple(tasks[task])
                task_results[task].append(result)

    if not args.combo_training:
        train_task(0, args.epochs)

        if args.ewc:
            train_loader, _ = get_dataset_loaders(tasks[0])
            elastic = ElasticConstraint(model_simple, train_loader, args)
        else:
            elastic = None

        train_task(1, args.epochs)

        train_task(2, args.epochs)

        train_task(3, args.epochs)
    else:
        train_task(4, args.epochs)

else:

    print("------Starting attention run------")

    #model_attention.use_entropy_loss(10 ** -5)

    def train_task(task_id, epochs):
        print("\n---Training on task " + tasks[task_id] + "---")

        for i in range(epochs):
            model_attention.set_task(task_id)
            train_attention(i + 1, tasks[task_id])

            for task in tasks:
                model_attention.set_task(task)
                result = test_attention(tasks[task])
                task_results[task].append(result)

    train_task(0, args.epochs)

    if args.ewc:
        train_loader, _ = get_dataset_loaders(tasks[0])
        elastic = ElasticConstraint(model_attention, train_loader, args)
    else:
        elastic = None

    #cummulated_ws = model_attention.get_w(0)
    #model_attention.set_cummulated_ws(cummulated_ws)
    #model_attention.use_l1_loss(10 ** 2)

    train_task(1, args.epochs)

s_MNIST = pd.Series(MNIST_results)
s_CIFAR10 = pd.Series(CIFAR10_results)
s_SVHN = pd.Series(SVHN_results)
s_FashionMNIST = pd.Series(FashionMNIST_results)
if args.combo_training:
    s_COMBO = pd.Series(COMBO_results)

timestamp = str(int(time.time()))
modifier = "Baseline simple on simultaneous " + str(args.epochs) + " epochs"

try:
    os.mkdir("results")
except:
    pass

try:
    os.mkdir("results/" + modifier)
except:
    pass

try:
    os.mkdir("results/" + modifier + "/" + timestamp)
except:
    pass

def get_filename_prefix():
    return "results/" + modifier + "/" + timestamp + "/"

s_MNIST.to_csv(get_filename_prefix() + "MNIST" +".csv")
s_CIFAR10.to_csv(get_filename_prefix() + "CIFAR10" + ".csv")
s_SVHN.to_csv(get_filename_prefix() + "SVHN" + ".csv")
s_FashionMNIST.to_csv(get_filename_prefix() + "FashionMNIST" + ".csv")
if args.combo_training:
    s_COMBO.to_csv(get_filename_prefix() + "COMBO" + ".csv")


s_MNIST.plot()
plt.savefig(get_filename_prefix() + "MNIST" +".png")
plt.clf()

s_CIFAR10.plot()
plt.savefig(get_filename_prefix() + "CIFAR10" + ".png")
plt.clf()

s_SVHN.plot()
plt.savefig(get_filename_prefix() + "SVHN" + ".png")
plt.clf()

s_FashionMNIST.plot()
plt.savefig(get_filename_prefix() + "FashionMNIST" + ".png")
plt.clf()

if args.combo_training:
    s_COMBO.plot()
    plt.savefig(get_filename_prefix() + "COMBO" + ".png")
    plt.clf()