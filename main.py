import torch, os
import numpy as np
from data.dataloader import DataProvider
import argparse
import datetime
import datetime
from config import create_config


def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    device = torch.device('cuda')

    ### Model config
    if args.model_name == "generic_metanet" or args.model_name == "resnet18_maml":
        from meta_maml import Meta
    # elif args.model_name == "metanet_maml_at":
    elif "maml_at" in args.model_name:
        from meta_maml_at import Meta
    elif args.model_name == "generic_protonet" or args.model_name == "resnet18_protonet":
        from meta_proto import Meta
    # elif args.model_name == "protonet_at":
    elif "protonet_at" in args.model_name:
        from meta_proto_at import Meta
    elif args.model_name == "nonfs_classification":
        from nonfs_classification_model import NonFSModel

    ### 1) Model
    if args.model_name == "generic_metanet" or args.model_name == "metanet_maml_at":
        config = create_config(num_layers=args.num_layers,
                               hidden_channel=args.hidden_channel, out_channel=args.n_way,
                               img_shape=args.img_shape, gate_rnr=False)
        model = Meta(args, config).to(device)
    elif args.model_name == "generic_protonet" or args.model_name == "protonet_at":
        config = create_config(num_layers=args.num_layers,
                               hidden_channel=args.hidden_channel, out_channel=args.emb_channel,
                               img_shape=args.img_shape, gate_rnr=False)
        model = Meta(args, config).to(device)
    elif args.model_name == "nonfs_classification":
        config = create_config(num_layers=args.num_layers,
                               hidden_channel=args.hidden_channel, out_channel=args.n_way,
                               img_shape=args.img_shape, gate_rnr=False)
        model = NonFSModel(args, config).to(device)

    print("---- Model config ----", config)
    # elif args.model_name == "resnet18_maml" or args.model_name == "resnet18_maml_at":
    #    # from config import config_maml_resnet_18
    #    config = None
    #    model = Meta(args, config).to(device)

    # elif args.model_name == "resnet18_protonet" or args.model_name == "resnet18_protonet_at":
    #    # from config import config_proto_resnet_18
    #    config = None
    #    model = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(model)
    print('Total trainable tensors:', num)

    ### 2) Dataloader
    dataloader = DataProvider(
        num_tasks=args.task_num,
        n_way=args.n_way,
        k_shot_spt=args.k_spt,
        k_shot_qry=args.k_qry,
    )

    if args.dataloader_mode == "few_shot":
        train_dl, validation_dl, test_dl = dataloader.load_few_shot_dataset(args.data_name)
    else:
        train_dl, validation_dl, test_dl = dataloader.load_dataset('omniglot')

    print(
        f"Dataset: {args.data_name}_{args.dataloader_mode}, training set: {len(train_dl)}, validation set: {len(validation_dl)}, testing set: {len(test_dl)}")
    ### 3) Training phase
    ##
    if args.mode == "train":
        current_datetime = datetime.datetime.now()
        datetime_string = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
        if args.store_dir is not None:
            store_dir = args.store_dir
        else:
            if args.adv_defense is None:
                store_dir = f"experiments/{args.data_name}/{args.model_name}/{datetime_string}"
            else:
                store_dir = f"experiments/{args.data_name}/{args.model_name}_{args.adv_defense}/{datetime_string}"
            os.makedirs(store_dir, exist_ok=True)
        ##
        acc_best = 0.0
        val_loss_list, val_acc_list, train_loss_list, train_acc_list = [], [], [], []
        val_loss_r_list, val_acc_r_list, train_loss_r_list, train_acc_r_list = [], [], [], []
        step = 0
        while step < args.epoch:
            val_loss, val_acc, train_loss, train_acc = 0.0, 0.0, 0.0, 0.0
            val_loss_r, val_acc_r, train_loss_r, train_acc_r = 0.0, 0.0, 0.0, 0.0

            for i, batch in enumerate(train_dl):
                if args.dataloader_mode == "few_shot":
                    x_spt, y_spt, x_qry, y_qry = batch
                    x_spt, y_spt, x_qry, y_qry = (
                        torch.stack(x_spt, dim=1).to(device),
                        torch.stack(y_spt, dim=1).to(device),
                        torch.stack(x_qry, dim=1).to(device),
                        torch.stack(y_qry, dim=1).to(device),
                    )
                    y_spt = y_spt.type(torch.LongTensor).to(device)
                    y_qry = y_qry.type(torch.LongTensor).to(device)

                    if "at" in args.model_name:
                        accs, loss, accs_r, loss_r = model(x_spt, y_spt, x_qry, y_qry)
                        train_loss += loss.detach().cpu().numpy()
                        train_acc += accs
                        train_loss_r += loss_r.detach().cpu().numpy()
                        train_acc_r += accs_r
                    else:
                        accs, loss = model(x_spt, y_spt, x_qry, y_qry)
                        train_loss += loss.detach().cpu().numpy()
                        train_acc += accs
                else:
                    x, y = batch
                    y = y.type(torch.LongTensor).to(device)
                    x, y = x.to(device), y.to(device)

                    accs, loss, accs_r, loss_r = model(x, y)
                    train_loss += loss.detach().cpu().numpy()
                    train_acc += accs
                    train_loss_r += loss_r.detach().cpu().numpy()
                    train_acc_r += accs_r

            train_loss /= args.task_num
            train_acc /= args.task_num
            train_loss_r /= args.task_num
            train_acc_r /= args.task_num

            log_str = f"Epoch: {step} Training Acc: {train_acc:.4f} Training Loss: {train_loss:.4f} Training Acc Robust: {train_acc_r:.4f} Training Loss Robust: {train_loss_r:.4f}\n"
            print(log_str)
            with open(f"{store_dir}/results.txt", "a") as f:
                f.writelines(log_str)
                f.close()

            if not os.path.exists(os.path.join(store_dir, "checkpoints")):
                os.makedirs(os.path.join(store_dir, "checkpoints"))

            # Saving training accs and loss
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            train_loss_r_list.append(train_loss_r)
            train_acc_r_list.append(train_acc_r)

            #### Validation ####
            val_loss_avg, val_acc_avg, val_loss_r_avg, val_acc_r_avg = 0.0, 0.0, 0.0, 0.0

            if args.dataloader_mode == "few_shot":
                for i, (x_spt, y_spt, x_qry, y_qry) in enumerate(validation_dl):
                    x_spt, y_spt, x_qry, y_qry = (
                        torch.stack(x_spt, dim=1).to(device),
                        torch.stack(y_spt, dim=1).to(device),
                        torch.stack(x_qry, dim=1).to(device),
                        torch.stack(y_qry, dim=1).to(device),
                    )
                    for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(
                            x_spt, y_spt, x_qry, y_qry):
                        if "at" in args.model_name:
                            accs, loss, accs_r, loss_r = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                            val_loss += loss.detach().cpu().numpy()
                            val_acc += accs
                            val_loss_r += loss_r.detach().cpu().numpy()
                            val_acc_r += accs_r
                        else:
                            accs, loss = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                            val_loss += loss.detach().cpu().numpy()
                            val_acc += accs
            else:
                for i, (x, y) in enumerate(validation_dl):
                    val_accs, val_loss, val_accs_r, val_loss_r = model.test(x, y)
                    val_loss_avg += val_loss
                    val_acc_avg += val_acc
                    val_loss_r_avg += val_loss_r
                    val_acc_r_avg += val_acc_r

                val_loss_avg /= args.task_num
                val_acc_avg /= args.task_num
                val_loss_r_avg /= args.task_num
                val_acc_r_avg /= args.task_num

                val_loss_list.append(val_loss_avg)
                val_acc_list.append(val_acc_avg)
                val_loss_r_list.append(val_loss_r_avg)
                val_acc_r_list.append(val_acc_avg)

                #### Save model ######
                params = {
                    'weight': model.state_dict(),
                    'epoch': step,
                    'train_loss': train_loss_list,
                    'train_acc': train_acc_list,
                    'train_loss_r': train_loss_r_list,
                    'train_acc_r': train_acc_r_list,
                    'val_loss': val_loss_list,
                    'val_acc': val_acc_list,
                    'val_loss_r': val_loss_r_list,
                    'val_acc_r': val_acc_r_list,
                }
                ######################
                torch.save(params, f"{store_dir}/checkpoints/last.pt")
                if val_acc_avg >= acc_best:
                    acc_best = val_acc_avg
                    print("Save best weights !!!")
                    torch.save(params, f"{store_dir}/checkpoints/best_new.pt")
                print(
                    "Epoch: {} Testing Acc: {:.4f}, Testing Loss: {:.4f}, Testing Acc Robust: {:.4f}, Testing Loss Robust: {:.4f} \n".format(
                        step, val_loss_avg, val_acc_avg, val_loss_r_avg, val_acc_r_avg)
                )
                with open(f"{store_dir}/results.txt", "a") as f:
                    f.writelines(
                        "Epoch: {} Testing Acc: {:.4f}, Testing Loss: {:.4f}, Testing Acc Robust: {:.4f}, Testing Loss Robust: {:.4f} \n".format(
                            step, val_loss_avg, val_acc_avg, val_loss_r_avg, val_acc_r_avg))
                    f.close()
            step += 1
    else:
        #### Testing ####
        model.load_state_dict(
            torch.load(args.weight))  ####################### load the best weight ##########################
        if args.adv_attack == "LinfPGD":
            from attack import PGD
            at = PGD(eps=args.adv_attack_eps, sigma=args.adv_attack_alpha, nb_iter=args.adv_attack_iters)
        else:
            at = None
        accs = []
        acc_test = 0.0
        acc_test_avg = 0.0
        for i, (x_spt, y_spt, x_qry, y_qry) in enumerate(test_dl):
            # test
            x_spt, y_spt, x_qry, y_qry = (
                torch.stack(x_spt, dim=1).to(device),
                torch.stack(y_spt, dim=1).to(device),
                torch.stack(x_qry, dim=1).to(device),
                torch.stack(y_qry, dim=1).to(device),
            )
            # print("--------------", x_spt.shape, x_qry.shape) # -------------- torch.Size([32, 5, 1, 28, 28]) torch.Size([32, 25, 1, 28, 28])

            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):

                # print("-----------------", x_spt_one.max(), x_qry_one.max()) # ----------------- tensor(1., device='cuda:0') tensor(1., device='cuda:0')

                if args.adv_attack == "LinfPGD":
                    # x_qry_one = at.attack(model.net, model.net.parameters(), x_qry_one, y_qry_one)

                    if args.model_name == "protonet_at" or args.model_name == "generic_protonet":
                        x_qry_one = at.attack(model.net, model.net.parameters(), x_qry_one, y_qry_one,
                                              extra_params=dict(x_spt=x_spt_one, y_spt=y_spt_one))
                    else:
                        x_qry_one = at.attack(model.net, model.net.parameters(), x_qry_one, y_qry_one)

                if args.model_name == "metanet_maml_at" or args.model_name == "resnet18_maml_at":
                    test_acc, accs_adv, accs_adv_prior = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                elif args.model_name == "generic_metanet" or args.model_name == "resnet18_maml":

                    test_acc = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                else:
                    test_loss, test_acc = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs.append(test_acc)
                # if args.model_name == "protonet_at" or args.model_name == "generic_protonet":
                if "protonet" in args.model_name:
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
    ### Model
    # Task 1: "generic_metanet" (1) "metanet_maml_at" (2) "generic_protonet" (3) "protonet_at" (4) "nonfs_classification" (5)
    # Task 2: "metanet_maml_at_gru" (5)  "protonet_at_gru" (6)
    argparser.add_argument('--model_name', type=str, help='The model name', default="nonfs_classification")
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1000)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)  # 0.001
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)  # 0.4
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--hidden_channel', type=int, help='hidden_channel', default=64)  # 64 128
    argparser.add_argument('--emb_channel', type=int, help='emb_channel', default=64)  # 64
    argparser.add_argument('--num_layers', type=int, help='The number of layers', default=2)
    argparser.add_argument('--weight', type=str, help='The learning phase', default=None)
    argparser.add_argument('--store_dir', type=str, help='The learning phase', default=None)

    ### Data
    argparser.add_argument('--mode', type=str, help='The learning phase', default="train")  # train test
    argparser.add_argument('--data_name', type=str, help='The data configuration',
                           default="cifar100")  # omniglot cifar10
    argparser.add_argument('--img_shape', type=tuple, help='The image shape',
                           default=(1, 28, 28))  # omniglot: (1, 28, 28)  cifar10: (3, 32, 32)
    argparser.add_argument('--dataloader_mode', type=str, help='dataloader mode',
                           default="non_few_shot")  # few_show, non_few_shot

    ### Learning phases
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)  # 1000 5000
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--train_period_print', type=int, help='train_period_print', default=1)
    argparser.add_argument('--test_period_print', type=int, help='test_period_print', default=5)
    argparser.add_argument('--test_size', type=int, help='test_size', default=5)
    argparser.add_argument('--weight_robust', type=float, help='weight_robust', default=1.0)

    ## Adversarial attack
    argparser.add_argument('--adv_attack_type', type=str, default="white_box",
                           help="The adversarial attack type")  # white_box, black_box
    argparser.add_argument('--adv_attack', type=str, default="LinfPGD",
                           help="The adversarial attack")  ### None LinfPGD FGSM
    argparser.add_argument('--adv_attack_eps', type=float, default=32 / 255,
                           help="The adversarial attack pertuabation level value")  # 8/255 16/255 32/255 64/255 128/255 1
    argparser.add_argument('--adv_attack_alpha', type=float, default=4 / 255,
                           help="The adversarial attack step size value")  # 4/255 16/255 32/255 64/255 128/255 1
    argparser.add_argument('--adv_attack_iters', type=float, default=7,
                           help="The adversarial attack number of iterations value")  # 7 9 11 13 17 19
    argparser.add_argument('--adv_defense', type=str, default=None, help="The adversarial defense")  ### None AT  TRADES
    ## Version control
    # argparser.add_argument('--leanrer_version', type=int, default=2, help="leanrer_version") # 1, 2
    args = argparser.parse_args()

    main(args)
