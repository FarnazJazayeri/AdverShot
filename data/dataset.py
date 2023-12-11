import torch
import numpy as np
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
            if hasattr(dataset, 'targets') and dataset.targets is not None:
                self.targets = torch.tensor(dataset.targets)
            else:
                self.targets = torch.tensor([y for _, y in dataset])
        else:
            self.targets = torch.tensor(targets)

        self.unique_targets = torch.unique(self.targets)
        self.shuffled_indices = torch.stack(
            [
                torch.randperm(len(self.unique_targets), generator=self.seed)[:self.n_way]
                for _ in range(self.size)
            ]
        )

        self.target_mapping = {
            t.item(): torch.arange(len(self.dataset))[self.targets == t] for t in self.unique_targets
        }

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        sampled_classes = self.unique_targets[self.shuffled_indices[idx]]

        support_set_x = []
        support_set_y = []
        query_set_x = []
        query_set_y = []
        for i, _class in enumerate(sampled_classes):
            class_samples = Subset(self.dataset, self.target_mapping[_class.item()])
            try:
                sample_indices = np.random.choice(range(len(class_samples)), self.k_sprt + self.k_query,
                                                  replace=False)
            except ValueError:
                # If number of samples are less that (k_spt + k_qry), then sample with replace=True
                sample_indices = np.random.choice(range(len(class_samples)), self.k_sprt + self.k_query,
                                                  replace=True)

            class_data = [class_samples[i] for i in sample_indices]
            class_data_x = [data[0] for data in class_data]
            class_data_y = [i for _ in class_data]
            support_set_x.extend(class_data_x[:self.k_sprt])
            support_set_y.extend(class_data_y[:self.k_sprt])
            query_set_x.extend(class_data_x[self.k_sprt:])
            query_set_y.extend(class_data_y[self.k_sprt:])

        return support_set_x, support_set_y, query_set_x, query_set_y

        # Randomly sample indices for support and query sets for chosen classes
        # sampled_indices = torch.stack(
        #     [
        #         (m := self.target_mapping[c])[
        #             torch.randperm(len(m), generator=self.seed)
        #         ][: self.k_sprt + self.k_query]
        #         for c in sampled_classes
        #     ]
        # )
        #
        # # Split the sampled indices into support and query indices
        # sprt_idx = sampled_indices[:, : self.k_sprt].reshape(-1)
        # query_idx = sampled_indices[:, self.k_sprt, :].reshape(-1)
        #
        # # Shuffle indices if requested
        # if self.shuffle:
        #     sprt_idx = sprt_idx[torch.randperm(len(sprt_idx), generator=self.seed)]
        #     query_idx = query_idx[torch.randperm(len(query_idx), generator=self.seed)]
        #
        # # Create Subset objects for support and query data
        # sprt_data = Subset(self.dataset, sprt_idx)
        # query_data = Subset(self.dataset, query_idx)
        #
        # # Reset labels to be in range [0, n_way] for each set
        # if self.reset_labels:
        #     sprt_data = [
        #         (x, torch.argwhere(sampled_classes == y).squeeze())
        #         for x, y in sprt_data
        #     ]
        #     query_data = [
        #         (x, torch.argwhere(sampled_classes == y).squeeze())
        #         for x, y in query_data
        #     ]
        #
        # return sprt_data, query_data

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
