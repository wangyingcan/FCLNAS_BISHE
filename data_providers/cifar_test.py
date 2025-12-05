# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
import torch
# import torch.utils.data.dataloader as DataLoader
from torch.utils.data import DataLoader, Dataset, sampler

import torch.utils.data
# import torch.utils.data.DataLoader as DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from data_providers.base_provider import *
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CifarDataProvider(DataProvider):

    def __init__(self, client_id=10, dataset_location=None, train_batch_size=1024,
                 test_batch_size=256, valid_size=None,
                 n_worker=2):
        self.client_id = client_id
        self.dset_save_path = dataset_location

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        dataset_class = datasets.CIFAR10

        train_transforms, test_transform = self.cifar_data_transforms()

        self.train_dataset = dataset_class(root=self.train_path,
                                           train=True,
                                           download=False,
                                           transform=train_transforms)
        self.test_dataset = dataset_class(root=self.valid_path,
                                          train=False,
                                          download=False,
                                          transform=test_transform)

        # [0,1,2] [3,4,5] [6,7,8,9]
        if self.client_id < 10:
            trn_set_list, test_set_list = self.get_image_idx(client=self.client_id)
            self.trn_set = torch.utils.data.Subset(self.train_dataset, indices=trn_set_list)
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, indices=test_set_list)
        else:
            self.trn_set = torch.utils.data.Subset(self.train_dataset, indices=range(50000))

        self.trn_set_length = len(self.trn_set)
        self.tst_set_length = len(self.test_dataset)
        self.n_worker = n_worker

    @staticmethod
    def name():
        return 'cifar'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self.dset_save_path is None:
            raise ValueError('unable to Use cifar10')
        return self.dset_save_path

    @property
    def data_url(self):
        raise ValueError('unable to Use cifar10')

    @property
    def train_path(self):
        return self.dset_save_path

    @property
    def valid_path(self):
        return self.dset_save_path

    def cifar_data_transforms(self, cutout_length=16):
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.2023, 0.1994, 0.2010]
        transf = [
            transforms.RandomCrop(32, padding=3),
            transforms.RandomHorizontalFlip()
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
        train_transform = transforms.Compose(transf + normalize)

        if cutout_length > 0:
            train_transform.transforms.append(Cutout(cutout_length))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        return train_transform, test_transform

    @property
    def resize_value(self):
        return 36

    @property
    def image_size(self):
        return 32

    def train(self):
        return DataLoaderX(
            self.trn_set,
            batch_size=self.train_batch_size,
            shuffle=True,  # sampler=self.train_sampler,
            num_workers=self.n_worker,
            pin_memory=True,
            drop_last=False,
        )

    def valid(self):
        return DataLoaderX(self.test_dataset,  # DatasetSplit(self.test_dataset, self.test_idxs),
                           batch_size=self.test_batch_size,
                           num_workers=self.n_worker,
                           pin_memory=True,
                           drop_last=False,
                           )

    def test(self):
        return DataLoaderX(self.test_dataset,  # DatasetSplit(self.test_dataset, self.test_idxs),
                           batch_size=self.test_batch_size,
                           num_workers=self.n_worker,
                           pin_memory=True,
                           drop_last=False,
                           )

    def get_image_idx(self, client=10):
        """
        return train_dataset, test_dataset, user_groups
        Get torchvision dataset
        [0,1,2] [3,4,5] [6,7,8,9]
        """
        idxs = np.arange(50000)
        labels = np.array(self.train_dataset.targets)

        test_idxs = np.arange(10000)
        test_labels = np.array(self.test_dataset.targets)

        # split trn/val
        trn_idxs = idxs[:50000]
        trn_labels = labels[:50000]

        if client < 3:
            # sort labels for trn
            trn_idxs_labels = np.vstack((trn_idxs, trn_labels))
            trn_idxs_labels = trn_idxs_labels[:, trn_idxs_labels[1, :] < 3]
            trn_idxs = trn_idxs_labels[0, :]

            # sort labels for test
            test_idxs_labels = np.vstack((test_idxs, test_labels))
            test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :] < 3]
            test_idxs = test_idxs_labels[0, :]

            if client == 0:
                trn_idxs = trn_idxs[:len(trn_idxs) // 3]
                test_idxs = test_idxs[:len(test_idxs) // 3]
            elif client == 1:
                trn_idxs = trn_idxs[len(trn_idxs) // 3:2 * len(trn_idxs) // 3]
                test_idxs = test_idxs[len(test_idxs) // 3:2 * len(test_idxs) // 3]
            else:
                trn_idxs = trn_idxs[2 * len(trn_idxs) // 3:]
                test_idxs = test_idxs[2 * len(test_idxs) // 3:]
        elif client > 5:
            # sort labels for trn
            trn_idxs_labels = np.vstack((trn_idxs, trn_labels))
            trn_idxs_labels = trn_idxs_labels[:, trn_idxs_labels[1, :] > 5]
            trn_idxs = trn_idxs_labels[0, :]

            # sort labels for test
            test_idxs_labels = np.vstack((test_idxs, test_labels))
            test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :] > 5]
            test_idxs = test_idxs_labels[0, :]

            if client == 6:
                trn_idxs = trn_idxs[:len(trn_idxs) // 4]
                test_idxs = test_idxs[:len(test_idxs) // 4]
            elif client == 7:
                trn_idxs = trn_idxs[len(trn_idxs) // 4:2 * len(trn_idxs) // 4]
                test_idxs = test_idxs[len(test_idxs) // 4:2 * len(test_idxs) // 4]
            elif client == 8:
                trn_idxs = trn_idxs[2 * len(trn_idxs) // 4:3 * len(trn_idxs) // 4]
                test_idxs = test_idxs[2 * len(test_idxs) // 4:3 * len(test_idxs) // 4]
            else:
                trn_idxs = trn_idxs[3 * len(trn_idxs) // 4:]
                test_idxs = test_idxs[3 * len(test_idxs) // 4:]
        else:
            # sort labels for trn
            trn_idxs_labels = np.vstack((trn_idxs, trn_labels))
            trn_idxs_labels = trn_idxs_labels[:, trn_idxs_labels[1, :] < 6]
            trn_idxs_labels = trn_idxs_labels[:, trn_idxs_labels[1, :] > 2]
            trn_idxs = trn_idxs_labels[0, :]

            # sort labels for test
            test_idxs_labels = np.vstack((test_idxs, test_labels))
            test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :] < 6]
            test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :] > 2]
            test_idxs = test_idxs_labels[0, :]

            if client == 3:
                trn_idxs = trn_idxs[:len(trn_idxs) // 3]
                test_idxs = test_idxs[:len(test_idxs) // 3]
            elif client == 4:
                trn_idxs = trn_idxs[len(trn_idxs) // 3:2 * len(trn_idxs) // 3]
                test_idxs = test_idxs[len(test_idxs) // 3:2 * len(test_idxs) // 3]
            else:
                trn_idxs = trn_idxs[2 * len(trn_idxs) // 3:]
                test_idxs = test_idxs[2 * len(test_idxs) // 3:]

        # shuffle trn/val index for to random
        np.random.shuffle(trn_idxs)
        np.random.shuffle(test_idxs)
        return trn_idxs.tolist(), test_idxs.tolist()

class CifarDataProvider100(DataProvider):

    def __init__(self, client_id=10, dataset_location=None, train_batch_size=1024,
                 test_batch_size=256, valid_size=None,
                 n_worker=2):
        self.client_id = client_id
        self.dset_save_path = dataset_location

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        dataset_class = datasets.CIFAR100

        train_transforms, test_transform = self.cifar_data_transforms()

        self.train_dataset = dataset_class(root=self.train_path,
                                           train=True,
                                           download=False,
                                           transform=train_transforms)
        self.test_dataset = dataset_class(root=self.valid_path,
                                          train=False,
                                          download=False,
                                          transform=test_transform)

        # [0,1,2] [3,4,5] [6,7,8,9]
        if self.client_id < 10:
            trn_set_list, test_set_list = self.get_image_idx(client=self.client_id)
            self.trn_set = torch.utils.data.Subset(self.train_dataset, indices=trn_set_list)
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, indices=test_set_list)
        else:
            self.trn_set = torch.utils.data.Subset(self.train_dataset, indices=range(50000))

        self.trn_set_length = len(self.trn_set)
        self.tst_set_length = len(self.test_dataset)
        self.n_worker = n_worker

    @staticmethod
    def name():
        return 'cifar100'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 100

    @property
    def save_path(self):
        if self.dset_save_path is None:
            raise ValueError('unable to Use cifar100')
        return self.dset_save_path

    @property
    def data_url(self):
        raise ValueError('unable to Use cifar100')

    @property
    def train_path(self):
        return self.dset_save_path

    @property
    def valid_path(self):
        return self.dset_save_path

    def cifar_data_transforms(self, cutout_length=8):
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.2023, 0.1994, 0.2010]
        transf = [
            transforms.RandomCrop(32, padding=3),
            transforms.RandomHorizontalFlip()
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
        train_transform = transforms.Compose(transf + normalize)

        if cutout_length > 0:
            train_transform.transforms.append(Cutout(cutout_length))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        return train_transform, test_transform

    @property
    def resize_value(self):
        return 36

    @property
    def image_size(self):
        return 32

    def train(self):
        return DataLoaderX(
            self.trn_set,
            batch_size=self.train_batch_size,
            shuffle=True,  # sampler=self.train_sampler,
            num_workers=self.n_worker,
            pin_memory=True,
            drop_last=False,
        )

    def valid(self):
        return DataLoaderX(self.test_dataset,  # DatasetSplit(self.test_dataset, self.test_idxs),
                           batch_size=self.test_batch_size,
                           num_workers=self.n_worker,
                           pin_memory=True,
                           drop_last=False,
                           )

    def test(self):
        return DataLoaderX(self.test_dataset,  # DatasetSplit(self.test_dataset, self.test_idxs),
                           batch_size=self.test_batch_size,
                           num_workers=self.n_worker,
                           pin_memory=True,
                           drop_last=False,
                           )

    def get_image_idx(self, client=10):
        """
        return train_dataset, test_dataset, user_groups
        Get torchvision dataset
        [0,1,2] [3,4,5] [6,7,8,9]
        """
        idxs = np.arange(50000)
        labels = np.array(self.train_dataset.targets)

        test_idxs = np.arange(10000)
        test_labels = np.array(self.test_dataset.targets)

        # split trn/val
        trn_idxs = idxs[:50000]
        trn_labels = labels[:50000]

        if client < 3:
            # sort labels for trn
            trn_idxs_labels = np.vstack((trn_idxs, trn_labels))
            trn_idxs_labels = trn_idxs_labels[:, trn_idxs_labels[1, :] < 33]
            trn_idxs = trn_idxs_labels[0, :]

            # sort labels for test
            test_idxs_labels = np.vstack((test_idxs, test_labels))
            test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :] < 33]
            test_idxs = test_idxs_labels[0, :]

            if client == 0:
                trn_idxs = trn_idxs[:len(trn_idxs) // 3]
                test_idxs = test_idxs[:len(test_idxs) // 3]
            elif client == 1:
                trn_idxs = trn_idxs[len(trn_idxs) // 3:2 * len(trn_idxs) // 3]
                test_idxs = test_idxs[len(test_idxs) // 3:2 * len(test_idxs) // 3]
            else:
                trn_idxs = trn_idxs[2 * len(trn_idxs) // 3:]
                test_idxs = test_idxs[2 * len(test_idxs) // 3:]
        elif client > 5:
            # sort labels for trn
            trn_idxs_labels = np.vstack((trn_idxs, trn_labels))
            trn_idxs_labels = trn_idxs_labels[:, trn_idxs_labels[1, :] > 66]
            trn_idxs = trn_idxs_labels[0, :]

            # sort labels for test
            test_idxs_labels = np.vstack((test_idxs, test_labels))
            test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :] > 66]
            test_idxs = test_idxs_labels[0, :]

            if client == 6:
                trn_idxs = trn_idxs[:len(trn_idxs) // 4]
                test_idxs = test_idxs[:len(test_idxs) // 4]
            elif client == 7:
                trn_idxs = trn_idxs[len(trn_idxs) // 4:2 * len(trn_idxs) // 4]
                test_idxs = test_idxs[len(test_idxs) // 4:2 * len(test_idxs) // 4]
            elif client == 8:
                trn_idxs = trn_idxs[2 * len(trn_idxs) // 4:3 * len(trn_idxs) // 4]
                test_idxs = test_idxs[2 * len(test_idxs) // 4:3 * len(test_idxs) // 4]
            else:
                trn_idxs = trn_idxs[3 * len(trn_idxs) // 4:]
                test_idxs = test_idxs[3 * len(test_idxs) // 4:]
        else:
            # sort labels for trn
            trn_idxs_labels = np.vstack((trn_idxs, trn_labels))
            trn_idxs_labels = trn_idxs_labels[:, trn_idxs_labels[1, :] < 67]
            trn_idxs_labels = trn_idxs_labels[:, trn_idxs_labels[1, :] > 32]
            trn_idxs = trn_idxs_labels[0, :]

            # sort labels for test
            test_idxs_labels = np.vstack((test_idxs, test_labels))
            test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :] < 67]
            test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :] > 32]
            test_idxs = test_idxs_labels[0, :]

            if client == 3:
                trn_idxs = trn_idxs[:len(trn_idxs) // 3]
                test_idxs = test_idxs[:len(test_idxs) // 3]
            elif client == 4:
                trn_idxs = trn_idxs[len(trn_idxs) // 3:2 * len(trn_idxs) // 3]
                test_idxs = test_idxs[len(test_idxs) // 3:2 * len(test_idxs) // 3]
            else:
                trn_idxs = trn_idxs[2 * len(trn_idxs) // 3:]
                test_idxs = test_idxs[2 * len(test_idxs) // 3:]

        # shuffle trn/val index for to random
        np.random.shuffle(trn_idxs)
        np.random.shuffle(test_idxs)
        return trn_idxs.tolist(), test_idxs.tolist()


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        # super(ChunkSampler, self).__init__()
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class DatasetSplit():
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        x = torch.tensor(image)
        y = torch.tensor(label)
        return x, y
