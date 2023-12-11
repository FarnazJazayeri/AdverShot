import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def get_dataset(name: str, *args, **kwargs) -> Dataset:
    kwargs.setdefault("download", True)
    kwargs.setdefault("transform", get_transforms(name))

    name = name.lower()
    if name == "omniglot":
        from torchvision.datasets import Omniglot

        return Omniglot(*args, **kwargs)
    elif name == "cifar10":
        from torchvision.datasets import CIFAR10

        return CIFAR10(*args, **kwargs)
    elif name == "cifar100":
        from torchvision.datasets import CIFAR100

        return CIFAR100(*args, **kwargs)
    elif name == "mnist":
        from torchvision.datasets import MNIST

        return MNIST(*args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {name}")


def get_transforms(name: str, *args, **kwargs) -> Compose:
    name = name.lower()
    if name == "omniglot":
        return Compose(
            [
                Resize((28, 28)),
                ToTensor(),
                Normalize((0.92206,), (0.08426,)),
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


class NShotDataset(Dataset):

    """
    Class to handle few-shot dataset
    Args:
    - dataset: base dataset
    - n_way: number of classes in a classification task
    - k_sprt: number of support examples per class
    - k_query: number of query examples per class
    - seed: random seed
    - targets: custom class targets
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        k_sprt: int,
        k_query: int,
        size: int = 1000,
        seed: int = None,
        targets=None,
        shuffle: bool = True,
        reset_labels: bool = True,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_sprt = k_sprt
        self.k_query = k_query
        self.size = size
        self.seed = torch.Generator()
        if seed is not None:
            self.seed.manual_seed(seed)
        self.shuffle = shuffle
        self.reset_labels = reset_labels

        if targets is None:
            self.targets = torch.tensor(dataset.targets)
        else:
            self.targets = torch.tensor(targets)

        self.unique_targets = torch.unique(self.targets)
        self.shuffled_indices = torch.stack(
            [
                torch.randperm(len(self.unique_targets), generator=self.seed)[
                    : self.n_way
                ]
                for _ in range(self.size)
            ]
        )

        self.target_mapping = torch.stack(
            [
                torch.arange(len(self.dataset))[self.targets == t]
                for t in self.unique_targets
            ]
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        sampled_classes = self.unique_targets[self.shuffled_indices[idx]]

        # Randomly sample indices for support and query sets for chosen classes
        sampled_indices = torch.stack(
            [
                (m := self.target_mapping[c])[
                    torch.randperm(len(m), generator=self.seed)
                ][: self.k_sprt + self.k_query]
                for c in sampled_classes
            ]
        )

        # Split the sampled indices into support and query indices
        sprt_idx = sampled_indices[:, : self.k_sprt].reshape(-1)
        query_idx = sampled_indices[:, self.k_sprt :].reshape(-1)

        # Shuffle indices if requested
        if self.shuffle:
            sprt_idx = sprt_idx[torch.randperm(len(sprt_idx), generator=self.seed)]
            query_idx = query_idx[torch.randperm(len(query_idx), generator=self.seed)]

        # Create Subset objects for support and query data
        sprt_data = Subset(self.dataset, sprt_idx)
        query_data = Subset(self.dataset, query_idx)

        # Reset labels to be in range [0, n_way] for each set
        if self.reset_labels:
            sprt_data = [
                (x, torch.argwhere(sampled_classes == y).squeeze())
                for x, y in sprt_data
            ]
            query_data = [
                (x, torch.argwhere(sampled_classes == y).squeeze())
                for x, y in query_data
            ]

        return sprt_data, query_data

    @staticmethod
    def get_collate_fn(input_to_tensor: bool = True, label_to_tensor: bool = False):
        def join_batch(data, return_label: bool = False):
            require_cast = label_to_tensor if return_label else input_to_tensor
            idx = 1 if return_label else 0
            if require_cast:
                return torch.stack(
                    [
                        torch.stack([torch.as_tensor(item[idx]) for item in group])
                        for group in data
                    ]
                )
            else:
                return [[item[idx] for item in group] for group in data]

        def collate_fn(batch):
            sprt_batch, query_batch = zip(*batch)

            sprt_data = join_batch(sprt_batch)
            sprt_labels = join_batch(sprt_batch, return_label=True)

            query_data = join_batch(query_batch)
            query_labels = join_batch(query_batch, return_label=True)

            return sprt_data, sprt_labels, query_data, query_labels

        return collate_fn
