from    omniglot import Omniglot
import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np


class DataLoader:

    def __init__(self, data_name, root, batchsz, n_way, k_shot, k_query, imgsz, num_episodes=10, train_percent=0.75):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.data_name = data_name
        self.resize = imgsz
        self.num_episodes = num_episodes
        if not os.path.isfile(os.path.join(root, f'{data_name}.npy')):
            # if root/data.npy does not exist, just download it
            if data_name == "omniglot":
                    self.x = Omniglot(root, download=True,
                                      transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                                    lambda x: x.resize((imgsz, imgsz)),
                                                                    lambda x: np.reshape(x, (imgsz, imgsz, 1)),
                                                                    lambda x: np.transpose(x, [2, 0, 1]),
                                                                    lambda x: x/255.]))
            else:
                raise ValueError("This dataset is not defined !!!")

            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for (img, label) in self.x:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.x = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                self.x.append(np.array(imgs))

            # as different class may have different number of imgs
            self.x = np.array(self.x).astype(float)  # [[20 imgs],..., 1623 classes in total]
            # each character contains 20 imgs
            print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, f'{data_name}.npy'), self.x)
            print(f'write into {data_name}.npy.')
        else:
            # if data.npy exists, just load it.
            self.x = np.load(os.path.join(root, f'{data_name}.npy'))
            print(f'load from {data_name}.npy.')

        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        self.x_train, self.x_test = self.x[:int(train_percent * self.x.shape[0])], self.x[int(train_percent * self.x.shape[0]):]

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(self.num_episodes):  # num of episodes

            x_spts, y_spts, y_cls_spts, x_qrys, y_qrys, y_cls_qrys = [], [], [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, y_cls_spt, x_qry, y_qry, y_cls_qry = [], [], [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False) # data_pack.shape[0] = 1200 (train), 423 (test)

                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(20, self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:self.k_shot + self.k_query]])
                    y_spt.append([j for _ in range(self.k_shot)]) ### 0 -> k-1
                    y_qry.append([j for _ in range(self.k_query)]) ### 0 -> k-1
                    y_cls_spt.append([cur_class for _ in range(self.k_shot)]) ### real classes
                    y_cls_qry.append([cur_class for _ in range(self.k_query)]) ### real classes

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                y_cls_spt = np.array(y_cls_spt).reshape(self.n_way * self.k_shot)[perm]
                #
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]
                y_cls_qry = np.array(y_cls_qry).reshape(self.n_way * self.k_query)[perm]
                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                y_cls_spts.append(y_cls_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)
                y_cls_qrys.append(y_cls_qry)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(int).reshape(self.batchsz, setsz)
            y_cls_spts = np.array(y_cls_spts).astype(int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(int).reshape(self.batchsz, querysz)
            y_cls_qrys = np.array(y_cls_qrys).astype(int).reshape(self.batchsz, querysz)
            data_cache.append([x_spts, y_spts, y_cls_spts, x_qrys, y_qrys, y_cls_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch


if __name__ == '__main__':

    import  time
    import  torch

    # plt.ion()
    #viz = visdom.Visdom(env='omniglot_view')

    db = DataLoader('omniglot', 'omniglot', batchsz=20, n_way=5, k_shot=5, k_query=15, imgsz=28)

    for i in range(1000):
        x_spt, y_spt, x_qry, y_qry = db.next('train')


        # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
        x_spt = torch.from_numpy(x_spt)
        x_qry = torch.from_numpy(x_qry)
        y_spt = torch.from_numpy(y_spt)
        y_qry = torch.from_numpy(y_qry)
        batchsz, setsz, c, h, w = x_spt.size()


