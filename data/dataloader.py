from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader, random_split

from data.dataset import NShotDataset

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale


class MyDataLoader:
    def __init__(self, num_tasks, n_way, k_shot_spt, k_shot_qry, train_split_perc=0.75, validation_split_perc=0.1):
        self.num_tasks = num_tasks
        self.n_way = n_way
        self.k_shot_spt = k_shot_spt
        self.k_shot_qry = k_shot_qry
        self.train_split_perc = train_split_perc
        self.valid_split_perc = validation_split_perc

    def load_few_shot_dataset(self, dataset_name):
        dataset = self._get_dataset(dataset_name)
        train, validation, test = self.make_few_shot_dataloaders(dataset)
        return train, validation, test

    def load_dataset(self, dataset_name):
        dataset = self._get_dataset(dataset_name)
        train_ds, validation_ds, test_ds = self._random_split(dataset)

        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        validation_dl = DataLoader(validation_ds, batch_size=64, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

        return train_dl, validation_dl, test_dl, [len(train_ds), len(validation_ds), len(test_ds)]

    def make_few_shot_dataloaders(self, dataset):
        train_dataset, validation_dataset, test_dataset = self._random_split(dataset)
        train_few_shot_dataset = self.make_few_shot_dataset(train_dataset, num_tasks=self.num_tasks)
        validation_few_shot_dataset = self.make_few_shot_dataset(validation_dataset, num_tasks=self.num_tasks)
        test_few_shot_dataset = self.make_few_shot_dataset(test_dataset, num_tasks=self.num_tasks)

        train_few_shot_dataloader = DataLoader(train_few_shot_dataset, batch_size=64, shuffle=True)
        validation_few_shot_dataloader = DataLoader(validation_few_shot_dataset, batch_size=64, shuffle=False)
        test_few_shot_dataloader = DataLoader(test_few_shot_dataset, batch_size=64, shuffle=False)

        return train_few_shot_dataloader, validation_few_shot_dataloader, test_few_shot_dataloader

    def make_few_shot_dataset(self, dataset, num_tasks=None):
        return NShotDataset(
            dataset=dataset,
            n_way=self.n_way,
            k_sprt=self.k_shot_spt,
            k_query=self.k_shot_qry,
            size=num_tasks or self.num_tasks,

        )

    def _get_dataset(self, name: str, *args, **kwargs) -> Dataset:
        kwargs.setdefault("download", True)
        kwargs.setdefault("transform", self._get_transforms(name))

        name = name.lower()
        if name == "omniglot":
            from torchvision.datasets import Omniglot

            return Omniglot('./dataset', *args, **kwargs)
        elif name == "cifar10":
            from torchvision.datasets import CIFAR10

            return CIFAR10('./dataset', *args, **kwargs)
        elif name == "cifar100":
            from torchvision.datasets import CIFAR100

            return CIFAR100('./dataset/', *args, **kwargs)
        elif name == "mnist":
            from torchvision.datasets import MNIST

            return MNIST('./dataset/', *args, **kwargs)
        else:
            raise ValueError(f"Unknown dataset {name}")

    def _get_transforms(self, name: str, *args, **kwargs) -> Compose:
        name = name.lower()
        if name == "omniglot":
            return Compose(
                [
                    Resize((28, 28)),
                    Grayscale(num_output_channels=1),
                    ToTensor()
                ]
            )
        elif name == "cifar10":
            return Compose(
                [
                    Resize((32, 32)),
                    ToTensor(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ]
            )
        elif name == "cifar100":
            return Compose(
                [
                    Resize((32, 32)),
                    ToTensor(),
                    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ]
            )
        elif name == "mnist":
            return Compose(
                [
                    Resize((28, 28)),
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,)),
                ]
            )
        else:
            return None

    def _random_split(self, dataset):
        train_size = int(self.train_split_perc * len(dataset))
        validation_size = int(self.valid_split_perc * len(dataset))
        test_size = len(dataset) - train_size - validation_size
        train_dataset, validation_dataset, test_dataset = random_split(dataset,
                                                                       [train_size, validation_size, test_size])

        return train_dataset, validation_dataset, test_dataset
