from torch.utils.data import ConcatDataset, DataLoader, sampler
from torchvision import transforms, datasets
data_folder = "./data"


def fminst_data():
    """ Get MNIST data """
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([.5, ], [.5, ])
         ])
    out_dir = data_folder
    train_data = datasets.FashionMNIST(root=out_dir, train=True, transform=compose, download=True)
    test_data = datasets.FashionMNIST(root=out_dir, train=False, transform=compose, download=True)
    return ConcatDataset([train_data, test_data])


def mnist_data():
    """ Get MNIST data """
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([.5, ], [.5, ])
         ])
    out_dir = data_folder
    train_data = datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
    test_data = datasets.MNIST(root=out_dir, train=False, transform=compose, download=True)
    return ConcatDataset([train_data, test_data])


def get_loader(batchSize=100, percentage=1, dataset="mnist"):
    if dataset == "mnist":
        data = mnist_data()
    elif dataset == "fmnist":
        data = fminst_data()
    else:
        raise Exception("dataset name not correct (or not implemented)")
    # get the size of updated data, based on percentage
    indices = [i for i in range(int(percentage * len(data)))]
    loader = DataLoader(data, batch_size=batchSize, sampler=sampler.SubsetRandomSampler(indices))
    return loader
