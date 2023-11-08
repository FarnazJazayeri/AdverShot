import torch
from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader, random_split

from data.few_shot import FewShotTaskDataset


class MyDataLoader:

    def __init__(self, num_tasks, n_way, k_shot_spt, k_shot_qry, train_split_perc=0.75, validation_split_perc=0.1):
        self.num_tasks = num_tasks
        self.n_way = n_way
        self.k_shot_spt = k_shot_spt
        self.k_shot_qry = k_shot_qry
        self.train_split_perc = train_split_perc
        self.valid_split_perc = validation_split_perc

    def load_few_shot_dataset(self, dataset_name):
        if dataset_name == 'omniglot':
            dataset = self.load_omniglot()
        elif dataset_name == 'mini_imagenet':
            dataset = self.load_mini_imagenet()
        elif dataset_name == 'cub':
            dataset = self.load_cub()
        else:
            return None

        train, validation, test = self.make_few_shot_task_dataloaders(dataset)
        return train, validation, test

    def load_omniglot(self):
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        omniglot_dataset = datasets.Omniglot(root='./dataset', download=True, transform=transform)
        return omniglot_dataset

    def load_mini_imagenet(self):
        # TODO
        return None

    def load_cub(self):
        # TODO
        return None

    def make_few_shot_task_dataloaders(self, dataset):
        train_size = int(self.train_split_perc * len(dataset))
        validation_size = int(self.valid_split_perc * len(dataset))
        test_size = len(dataset) - train_size - validation_size
        train_dataset, validation_dataset, test_dataset = random_split(dataset,
                                                                       [train_size, validation_size, test_size])
        train_few_shot_dataset = self.make_few_shot(train_dataset)
        validation_few_shot_dataset = self.make_few_shot(validation_dataset, num_tasks=200)
        test_few_shot_dataset = self.make_few_shot(test_dataset, num_tasks=200)

        train_few_shot_dataloader = DataLoader(train_few_shot_dataset, batch_size=64, shuffle=True)
        validation_few_shot_dataloader = DataLoader(validation_few_shot_dataset, batch_size=64, shuffle=False)
        test_few_shot_dataloader = DataLoader(test_few_shot_dataset, batch_size=64, shuffle=False)

        return train_few_shot_dataloader, validation_few_shot_dataloader, test_few_shot_dataloader

    def make_few_shot(self, dataset, num_tasks=None):
        return FewShotTaskDataset(
            dataset=dataset,
            num_tasks=num_tasks or self.num_tasks,
            n_way=self.n_way,
            k_shot_spt=self.k_shot_spt,
            k_shot_qry=self.k_shot_qry,
        )
