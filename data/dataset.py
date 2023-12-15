from functools import lru_cache
import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


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
        cache: bool = True,
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

        if cache:
            self._get_sprt_qry_idx = lru_cache(maxsize=None)(self._get_sprt_qry_idx)

        if targets is None:
            if hasattr(dataset, "targets"):
                self.targets = torch.as_tensor(dataset.targets)
            elif hasattr(dataset, "labels"):
                self.targets = torch.as_tensor(dataset.labels)
            else:
                self.targets = torch.as_tensor([item[1] for item in dataset])
        else:
            self.targets = torch.as_tensor(targets)

        self.unique_targets = torch.unique(self.targets)
        self.shuffled_indices = torch.stack(
            [
                torch.randperm(len(self.unique_targets), generator=self.seed)[
                    : self.n_way
                ]
                for _ in range(self.size)
            ]
        )

        self.per_target_data = torch.stack(
            [
                torch.arange(len(self.dataset))[self.targets == t]
                for t in self.unique_targets
            ]
        )

    def __len__(self) -> int:
        return self.size

    def _get_sprt_qry_idx(self, idx):
        sampled_classes = self.unique_targets[self.shuffled_indices[idx]]

        # Randomly sample indices for support and query sets for chosen classes
        sampled_indices = torch.stack(
            [
                data[torch.randperm(len(data), generator=self.seed)][
                    : self.k_sprt + self.k_query
                ]
                for data in self.per_target_data[sampled_classes]
            ]
        )

        # Split the sampled indices into support and query indices
        sprt_idx = sampled_indices[:, : self.k_sprt].reshape(-1)
        query_idx = sampled_indices[:, self.k_sprt :].reshape(-1)

        # Shuffle indices if requested
        if self.shuffle:
            sprt_idx = sprt_idx[torch.randperm(len(sprt_idx), generator=self.seed)]
            query_idx = query_idx[torch.randperm(len(query_idx), generator=self.seed)]

        return (sprt_idx, query_idx), sampled_classes

    def __getitem__(self, i: int):
        (sprt_idx, query_idx), sampled_classes = self._get_sprt_qry_idx(i)

        # Extract support and query data directly without creating Subset objects
        sprt_data = list(map(self.dataset.__getitem__, sprt_idx))
        query_data = list(map(self.dataset.__getitem__, query_idx))

        # Reset labels to be in range [0, n_way] for each set
        if self.reset_labels:
            # Create a mapping from original class indices to new indices
            class_map = {
                original: new for new, original in enumerate(sampled_classes.tolist())
            }

            # Apply the mapping to the labels
            sprt_data = [(x, class_map[y]) for x, y in sprt_data]
            query_data = [(x, class_map[y]) for x, y in query_data]

        return sprt_data, query_data

    @staticmethod
    def get_collate_fn(input_to_tensor: bool = True, label_to_tensor: bool = True):
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
