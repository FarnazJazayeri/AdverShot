import numpy as np
from torch.utils.data import Dataset, Subset


class FewShotTaskDataset(Dataset):
    def __init__(self, dataset, num_tasks, n_way, k_shot_spt, k_shot_qry):
        self.dataset = dataset
        self.num_tasks = num_tasks
        self.n_way = n_way
        self.k_shot_spt = k_shot_spt
        self.k_shot_qry = k_shot_qry
        self.classes = self.generate_class_mapping()

    def generate_class_mapping(self):
        mapping = {}
        for i, (x, y) in enumerate(self.dataset):
            if y not in mapping:
                mapping[y] = []
            mapping[y].append(i)
        return mapping

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        selected_classes = np.random.choice(list(self.classes.keys()), self.n_way, replace=False)

        # Collect support and query set samples
        support_set_x = []
        support_set_y = []
        query_set_x = []
        query_set_y = []
        for i, _class in enumerate(selected_classes):
            class_samples = Subset(self.dataset, self.classes[_class])
            sample_indices = np.random.choice(range(len(class_samples)), self.k_shot_spt + self.k_shot_qry,
                                              replace=False)
            class_data = [class_samples[i] for i in sample_indices]
            class_data_x = [data[0] for data in class_data]
            class_data_y = [i for _ in class_data]
            support_set_x.extend(class_data_x[:self.k_shot_spt])
            support_set_y.extend(class_data_y[:self.k_shot_spt])
            query_set_x.extend(class_data_x[self.k_shot_spt:])
            query_set_y.extend(class_data_y[self.k_shot_spt:])

        return support_set_x, support_set_y, query_set_x, query_set_y
