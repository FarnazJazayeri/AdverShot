import torch, os
import numpy as np
import argparse

from data.dataloader import MyDataLoader
from model.learner import Learner
from model.meta import Meta
from tools.attacks.white_box.pgd import PGD

device = torch.device('cuda:0')

def main(args):
    torch.manual_seed(222)
    # torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    device = torch.device('cuda:0')
    learner = Learner(
        config=config,
        imgc=1,
        imgsz=28,
    )
    attacker = PGD()
    maml = Meta(learner=learner,
                update_lr=0.01,
                meta_lr=0.01,
                adv_reg_param=0.001,
                update_steps=100,
                update_steps_test=100,
                attacker=attacker,
                device='cuda:0',
                ).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    dl = MyDataLoader(num_tasks=500,
                      n_way=5,
                      k_shot_spt=1,
                      k_shot_qry=1,
                      )

    train_dl, validation_dl, test_dl = dl.load_few_shot_dataset('omniglot')

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_dl):

        if step >= args.epoch:
            break
        # x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
        #                              torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        final_loss, losses_q, losses_q_adv, accs, accs_adv = maml(x_spt, y_spt, x_qry, y_qry)

        if step % 5 == 0:
            print('step:', step, '\ttraining acc:', accs)

        # if step % 500 == 0:
        #     accs = []
        #     for _ in range(1000//args.task_num):
        # test
        # x_spt, y_spt, x_qry, y_qry = test_dl.next()
        # x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
        #                              torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # split to single task each time
        # for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
        #     test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
        #     accs.append(test_acc)

        # [b, update_step+1]
        # accs = np.array(accs).mean(axis=0).astype(np.float16)
        # print('Test acc:', accs)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main(args)
