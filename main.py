import torch, os
import numpy as np
from omniglotNShot import OmniglotNShot
import argparse
import datetime
from meta import Meta
import datetime


def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    device = torch.device('cuda')

    ### Model config
    if args.model_name == "generic_metanet":
        from meta import Meta
    elif args.model_name == "metanet_maml_at":
        from meta_adv import Meta
    elif args.model_name == "generic_protonet":
        from meta_proto import Meta

    ### 1) Model
    if args.model_name == "generic_metanet" or args.model_name == "metanet_maml_at":
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
        model = Meta(args, config).to(device)
    elif args.model_name == "generic_protonet":
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
            ('linear', [64, 64])
        ]

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(model)
    print('Total trainable tensors:', num)

    ### 2) Train dataloader
    db_train = OmniglotNShot('omniglot',
                             batchsz=args.task_num,
                             n_way=args.n_way,
                             k_shot=args.k_spt,
                             k_query=args.k_qry,
                             imgsz=args.imgsz)

    ### 3) Training phase
    ##
    if args.mode == "train":
        current_datetime = datetime.datetime.now()
        datetime_string = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
        if args.adv_defense is None:
            store_dir = f"experiments/{args.data_name}/{args.model_name}/{datetime_string}"
        else:
            store_dir = f"experiments/{args.data_name}/{args.model_name}_{args.adv_defense}/{datetime_string}"
        os.makedirs(store_dir, exist_ok=True)
        ##
        acc_best = 0.0
        acc_test = 0.0
        for step in range(args.epoch):

            x_spt, y_spt, x_qry, y_qry = db_train.next()
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

            # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
            if args.model_name == "metanet_maml_at":
                accs, accs_adv = model(x_spt, y_spt, x_qry, y_qry)
            else:
                accs = model(x_spt, y_spt, x_qry, y_qry)

            if step % 10 == 0:
                print('Epoch: {} Training Acc: {}\n'.format(step, accs))
                with open(f'{store_dir}/results.txt', 'a') as f:
                    f.writelines('Epoch: {} Training Acc: {}\n'.format(step, accs))
                    f.close()

            ##
            if not os.path.exists(os.path.join(store_dir, 'checkpoints')):
                os.makedirs(os.path.join(store_dir, 'checkpoints'))
            ##

            #### Testing ####
            if step % 100 == 0:
                torch.save(model.state_dict(), f'{store_dir}/checkpoints/last.pt')
                accs = []
                for _ in range(1000 // args.task_num):
                    # test
                    x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                        torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                    # split to single task each time
                    for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                        if args.model_name == "metanet_maml_at":
                            test_acc, accs_adv, accs_adv_prior = model.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        else:
                            test_acc = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        accs.append(test_acc)
                        acc_test += test_acc[-1]
                    acc_test /= (1000 // args.task_num)
                # [b, update_step+1]

                # deep copy the model
                # print(type(accs), accs)
                if acc_test >= acc_best:
                    acc_best = acc_test
                    print("Save best weights !!!")
                    torch.save(model.state_dict(), f'{store_dir}/checkpoints/best_new.pt')
                acc_test = 0.0
                accs = np.array(accs).mean(axis=0).astype(np.float16)
                print('Epoch: {} Testing Acc: {}, Best Acc: {} \n'.format(step, accs, acc_best))
                with open(f'{store_dir}/results.txt', 'a') as f:
                    f.writelines('Epoch: {} Testing Acc: {}, Best Acc: {} \n'.format(step, accs, acc_best))
                    f.close()
    else:
        #### Testing ####
        model.load_state_dict(torch.load(args.weight))
        if args.adv_attack == "LinfPGD":
            from attack import PGD
            at = PGD(eps=args.adv_attack_eps, sigma=args.adv_attack_alpha, nb_iter=args.adv_attack_iters)
        else:
            at = None
        accs = []
        acc_test = 0.0
        for i in range(1000 // args.task_num):
            # test
            x_spt, y_spt, x_qry, y_qry = db_train.next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs.append(test_acc)
                acc_test += test_acc[-1]
            acc_test /= (1000 // args.task_num)
            print(f"Epoch: {i}, test acc: {acc_test}")
            with open(f"{os.path.dirname(args.weight)}/results_test.txt", 'a') as f:
                f.writelines(f"Epoch: {i}, test acc: {acc_test}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    ###
    argparser.add_argument('--mode', type=str, help='The learning phase', default="train")
    argparser.add_argument('--weight', type=str, help='The learning phase', default="/home/qle/Project/MetaLearning_FewShotLearning/source/MAML-Pytorch/experiments/omniglot/generic_metanet/2023-11-15_18-48-06/checkpoints/best_new.pt")
    argparser.add_argument('--data_name', type=str, help='The data configuration', default="omniglot")
    argparser.add_argument('--model_name', type=str, help='The model name', default="metanet_maml_at")  # "generic_protonet"  "generic_metanet" "metanet_maml_at"
    ## Adversarial attack
    argparser.add_argument('--adv_attack_type', type=str, default="white_box", help="The adversarial attack type")  # white_box, black_box
    argparser.add_argument('--adv_attack', type=str, default="LinfPGD", help="The adversarial attack")  ### LinfPGD FGSM
    argparser.add_argument('--adv_attack_eps', type=float, default=16 / 255, help="The adversarial attack pertuabation level value")  # 8/255 16/255 32/255 64/255 128/255 1
    argparser.add_argument('--adv_attack_alpha', type=float, default=4 / 255, help="The adversarial attack step size value")  # 4/255 16/255 32/255 64/255 128/255 1
    argparser.add_argument('--adv_attack_iters', type=float, default=7, help="The adversarial attack number of iterations value")  # 7 9 11 13 17 19
    argparser.add_argument('--adv_defense', type=str, default=None, help="The adversarial defense")  ### None AT  TRADES

    args = argparser.parse_args()

    main(args)
