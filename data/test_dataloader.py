from dataloader import DataLoader
import torch



#dataset = Omniglot("omniglot", download=False,
#            transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
#                                          lambda x: x.resize((28, 28)),
#                                          lambda x: np.reshape(x, (28, 28, 1)),
#                                          lambda x: np.transpose(x, [2, 0, 1]),
#                                          lambda x: x/255.]))
#print(len(dataset)) # 32460
# Total: 1623 classes (characters)
# 4 different persons => 1623 * 4 subclasses
#for i, batch in enumerate(dataset):
#    print(i, batch[0].shape, batch[1]) # 32459 (1, 28, 28) 1622

# (self, data_name, root, batchsz, n_way, k_shot, k_query, imgsz, num_episodes=10, train_percent=0.75)
db = DataLoader(data_name='omniglot', root='omniglot', batchsz=20, n_way=5, k_shot=15, k_query=10, imgsz=28, num_episodes=10, train_percent=0.75)
for i in range(1000):
    x_spt, y_spt, y_cls_spt, x_qry, y_qry, y_cls_qry = db.next('train')
    # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
    x_spt = torch.from_numpy(x_spt)
    x_qry = torch.from_numpy(x_qry)
    y_spt = torch.from_numpy(y_spt)
    y_qry = torch.from_numpy(y_qry)
    y_cls_spt = torch.from_numpy(y_cls_spt)
    y_cls_qry = torch.from_numpy(y_cls_qry)
    batchsz, setsz, c, h, w = x_spt.size()
    print(x_spt.shape, x_qry.shape) # torch.Size([20, 25, 1, 28, 28]) torch.Size([20, 75, 1, 28, 28])
    #print(y_cls_spt.shape, y_cls_qry.shape) # batchsz x n_way * k_shot
    #print(y_cls_spt.min(), y_cls_qry.min())
   # print(y_cls_spt.max(), y_cls_qry.max())

