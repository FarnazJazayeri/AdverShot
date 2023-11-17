import torch, os
import numpy as np
from omniglotNShot import OmniglotNShot
import argparse
import datetime
import datetime


def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    device = torch.device('cuda')

    ### Model config
    if args.model_name == "generic_metanet":
        from meta_maml import Meta
    elif args.model_name == "metanet_maml_at":
        from meta_maml_at import Meta
    elif args.model_name == "generic_protonet":
        from meta_proto import Meta
    elif args.model_name == "protonet_at":
        from meta_proto_at import Meta

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
    elif args.model_name == "generic_protonet" or args.model_name == "protonet_at":
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
        model = Meta(args, config).to(device)

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
            elif args.model_name == "generic_protonet" or args.model_name == "protonet_at":
                loss, accs = model(x_spt, y_spt, x_qry, y_qry)
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
                acc_test = 0.0
                torch.save(model.state_dict(), f'{store_dir}/checkpoints/last.pt')
                accs = []
                acc_test_avg = 0.0
                for _ in range(100):
                    # test
                    x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                        torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
                    # print("------------------------", x_spt.shape) # torch.Size([32, 5, 1, 28, 28])
                    # split to single task each time
                    for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                        # print("------------------------", x_spt_one.shape) # torch.Size([5, 1, 28, 28])
                        if args.model_name == "metanet_maml_at":
                            test_acc, accs_adv, accs_adv_prior = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        elif args.model_name == "generic_metanet":
                            test_acc = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        else:
                            test_loss, test_acc = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        accs.append(test_acc)
                        if args.model_name == "protonet_at" or args.model_name == "generic_protonet":
                            acc_test += test_acc
                        else:
                            acc_test += test_acc[-1]
                    acc_test /= args.task_num
                    acc_test_avg += acc_test
                # [b, update_step+1]
                acc_test_avg /= 100
                # print(type(accs), accs)
                if acc_test >= acc_best:
                    acc_best = acc_test
                    print("Save best weights !!!")
                    torch.save(model.state_dict(), f'{store_dir}/checkpoints/best_new.pt')
                    # accs = np.array(accs).mean(axis=0).astype(np.float16)
                print('Epoch: {} Testing Acc: {}, Best Acc: {} \n'.format(step, acc_test, acc_test_avg))
                with open(f'{store_dir}/results.txt', 'a') as f:
                    f.writelines('Epoch: {} Testing Acc: {}, Best Acc: {} \n'.format(step, acc_test, acc_test_avg))
                    f.close()
                acc_test = 0.0
    else:
        #### Testing ####
        model.load_state_dict(torch.load(args.weight))  ####################### load the best weight ##########################
        if args.adv_attack == "LinfPGD":
            from attack import PGD
            at = PGD(eps=args.adv_attack_eps, sigma=args.adv_attack_alpha, nb_iter=args.adv_attack_iters)
        else:
            at = None
        accs = []
        acc_test = 0.0
        acc_test_avg = 0.0
        for i in range(100):
            # test
            x_spt, y_spt, x_qry, y_qry = db_train.next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

            # print("--------------", x_spt.shape, x_qry.shape) # -------------- torch.Size([32, 5, 1, 28, 28]) torch.Size([32, 25, 1, 28, 28])

            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                # print("-----------------", x_spt_one.max(), x_qry_one.max()) # ----------------- tensor(1., device='cuda:0') tensor(1., device='cuda:0')

                if args.adv_attack == "LinfPGD":
                    # x_qry_one = at.attack(model.net, model.net.parameters(), x_qry_one, y_qry_one)
                    if args.model_name == "protonet_at" or args.model_name == "generic_protonet":
                        at.attack(model.net, model.net.parameters(), x_qry_one, y_qry_one,
                                  extra_params=dict(x_spt=x_spt_one, y_spt=y_spt_one))
                    else:
                        x_qry_one = at.attack(model.net, model.net.parameters(), x_qry_one, y_qry_one)

                if args.model_name == "metanet_maml_at":
                    test_acc, accs_adv, accs_adv_prior = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                elif args.model_name == "generic_metanet":
                    test_acc = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                else:
                    test_loss, test_acc = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs.append(test_acc)
                if args.model_name == "protonet_at" or args.model_name == "generic_protonet":
                    acc_test += test_acc
                else:
                    acc_test += test_acc[-1]
            acc_test /= args.task_num
            acc_test_avg += acc_test

            print(f"Epoch: {i}, test acc: {acc_test}")
            with open(f"{os.path.dirname(args.weight)}/results_test.txt", 'a') as f:
                f.writelines(f"Epoch: {i}, test acc: {acc_test} \n")
            acc_test = 0.0
        acc_test_avg /= 100
        print(f"Test acc avg: {acc_test_avg}")
        with open(f"{os.path.dirname(args.weight)}/results_test.txt", 'a') as f:
            f.writelines(f"Test acc avg: {acc_test_avg}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=5000)  # 1000 5000
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
    argparser.add_argument('--mode', type=str, help='The learning phase', default="train")  # train test
    argparser.add_argument('--weight', type=str, help='The learning phase', default="/home/qle/Project/MetaLearning_FewShotLearning/source/MAML-Pytorch/experiments/omniglot/generic_metanet/2023-11-16_23-08-34/checkpoints/best_new.pt")
    argparser.add_argument('--data_name', type=str, help='The data configuration', default="omniglot")
    argparser.add_argument('--model_name', type=str, help='The model name', default="protonet_at")  # "generic_metanet" (1) "metanet_maml_at" (2) "generic_protonet" (3) "protonet_at" (4)
    ## Adversarial attack
    argparser.add_argument('--adv_attack_type', type=str, default="white_box", help="The adversarial attack type")  # white_box, black_box
    argparser.add_argument('--adv_attack', type=str, default="LinfPGD", help="The adversarial attack")  ### None LinfPGD FGSM
    argparser.add_argument('--adv_attack_eps', type=float, default=16 / 255, help="The adversarial attack pertuabation level value")  # 8/255 16/255 32/255 64/255 128/255 1
    argparser.add_argument('--adv_attack_alpha', type=float, default=4 / 255, help="The adversarial attack step size value")  # 4/255 16/255 32/255 64/255 128/255 1
    argparser.add_argument('--adv_attack_iters', type=float, default=7, help="The adversarial attack number of iterations value")  # 7 9 11 13 17 19
    argparser.add_argument('--adv_defense', type=str, default=None, help="The adversarial defense")  ### None AT  TRADES

    args = argparser.parse_args()

    main(args)
