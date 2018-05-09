from __future__ import print_function
import torch
from torchvision import datasets, transforms

datasets_loaded = False
train_loaders, test_loaders = None, None

def load_datasets(args):
    global datasets_loaded, train_loaders, test_loaders

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    MNIST_train_dataset = datasets.MNIST('../data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.Pad(2),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]))
    MNIST_test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                                    transforms.Pad(2),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ]))

    CIFAR10_train_dataset = datasets.CIFAR10('../data', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.Grayscale(),
                                             #transforms.RandomCrop(32, padding=4),
                                             #transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                (0.2023, 0.1994, 0.2010))
                                         ]))
    CIFAR10_test_dataset = datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                            (0.2023, 0.1994, 0.2010))
                                        ]))

    COMBO_train_dataset = torch.utils.data.ConcatDataset(
                            [MNIST_train_dataset,
                            CIFAR10_train_dataset])
    COMBO_test_dataset = torch.utils.data.ConcatDataset(
                            [MNIST_test_dataset,
                            CIFAR10_test_dataset])

    get_dataloader = lambda dataset:  torch.utils.data.DataLoader(
                                    dataset, batch_size=args.batch_size,
                                    shuffle=True, **kwargs)

    train_loader_MNIST = get_dataloader(MNIST_train_dataset)
    test_loader_MNIST = get_dataloader(MNIST_test_dataset)

    train_loader_CIFAR10 = get_dataloader(CIFAR10_train_dataset)
    test_loader_CIFAR10 = get_dataloader(CIFAR10_test_dataset)

    train_loader_COMBO = get_dataloader(COMBO_train_dataset)
    test_loader_COMBO = get_dataloader(COMBO_test_dataset)


    train_loaders = {"MNIST": train_loader_MNIST,
                    "CIFAR10": train_loader_CIFAR10,
                    "COMBO": train_loader_COMBO}
    test_loaders = {"MNIST": test_loader_MNIST,
                    "CIFAR10": test_loader_CIFAR10,
                    "COMBO": test_loader_COMBO}

    datasets_loaded = True

def get_dataset_loaders(dataset_name):
    if not datasets_loaded:
        print("Datasets not loaded")
        exit()
    train_loader = train_loaders[dataset_name]
    test_loader = test_loaders[dataset_name]

    return train_loader, test_loader