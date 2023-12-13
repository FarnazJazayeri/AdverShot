import torch, os
import numpy as np
from data.dataloader import MyDataLoader
import argparse
import datetime
from config import create_config


def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    device = torch.device('cuda:0')

    ### Model config
    if args.model_name == "generic_metanet" or args.model_name == "resnet18_maml":
        from meta_maml import Meta

    elif "maml_at" in args.model_name:
        from meta_maml_at import Meta

    elif args.model_name == "generic_protonet" or args.model_name == "resnet18_protonet":
        from meta_proto import Meta

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
        model = NonFSModel(args, config, False).to(device)

    elif args.model_name == "nonfs_classification_at":
        config = create_config(num_layers=args.num_layers,
                               hidden_channel=args.hidden_channel, out_channel=args.n_way,
                               img_shape=args.img_shape, gate_rnr=False)
        model = NonFSModel(args, config, True).to(device)

    print("---- Model config ----", config)

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(model)
    print('Total trainable tensors:', num)

    ### 2) Dataloader
    dataloader = MyDataLoader(
        num_tasks=args.task_num,
        n_way=args.n_way,
        k_shot_spt=args.k_spt,
        k_shot_qry=args.k_qry,
    )
    if args.data_name == "omniglot":
        if args.dataloader_mode == "few_shot":
            train_dl, validation_dl, test_dl = dataloader.load_few_shot_dataset('omniglot')
        else:
            train_dl, validation_dl, test_dl = dataloader.load_dataset('omniglot')

    elif args.data_name == "cifar10":
        if args.dataloader_mode == "few_shot":
            train_dl, validation_dl, test_dl = dataloader.load_few_shot_dataset('cifar10')
        else:
            train_dl, validation_dl, test_dl, data_len = dataloader.load_dataset('cifar10')
    print(f"Dataset: {args.data_name}_{args.dataloader_mode}, training set: {len(train_dl)}, validation set: {len(validation_dl)}, testing set: {len(test_dl)}")
    ### 3) Training phase
    if args.mode == "train":
        current_datetime = datetime.datetime.now()
        datetime_string = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
        if args.store_dir is not None:
            store_dir = args.store_dir
        else:
            if "_at" not in args.model_name:
                store_dir = f"experiments/{args.data_name}/{args.model_name}/l{args.num_layers}_h{args.hidden_channel}"
            else:
                store_dir = f"experiments/{args.data_name}/{args.model_name}_{args.adv_defense}/l{args.num_layers}_h{args.hidden_channel}_r{str(args.weight_robust)}"
            os.makedirs(store_dir, exist_ok=True)

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
                else:
                    x, y = batch
                    x = x.to(device)
                    y = y.type(torch.LongTensor).to(device)

                if args.model_name == "metanet_maml_at" or args.model_name == "protonet_at":
                    accs, loss, accs_r, loss_r = model(x_spt, y_spt, x_qry, y_qry)
                    train_loss += loss.detach().cpu().numpy()
                    train_acc += accs
                    train_loss_r += loss_r.detach().cpu().numpy()
                    train_acc_r += accs_r
                elif args.model_name == "nonfs_classification" or args.model_name == "nonfs_classification_at":
                    accs, loss, accs_r, loss_r = model(x, y)
                    train_loss += loss.detach().cpu().numpy()
                    train_acc += accs
                    train_loss_r += float(loss_r.detach().cpu().numpy())
                    train_acc_r += accs_r
                else:
                    accs, loss = model(x_spt, y_spt, x_qry, y_qry)
                    train_loss += loss.detach().cpu().numpy()
                    train_acc += accs

            if args.dataloader_mode == "few_shot":
                train_loss /= (i + 1)
                train_acc /= (i + 1)
                train_loss_r /= (i + 1)
                train_acc_r /= (i + 1)
            else:
                print(i)
                print(data_len[0])
                train_loss /= data_len[0]
                train_acc /= data_len[0]
                train_loss_r /= data_len[0]
                train_acc_r /= data_len[0]
            if (step + 1) % args.train_period_print == 0:
                print("Epoch: {} Training Acc: {:.4f} Training Loss: {:.4f} Training Acc Robust: {:.4f} Training Loss Robust: {:.4f}\n".format(step, train_acc, train_loss, train_acc_r, train_loss_r))
                with open(f"{store_dir}/results.txt", "a") as f:
                    f.writelines("Epoch: {} Training Acc: {:.4f} Training Loss: {:.4f} Training Acc Robust: {:.4f} Training Loss Robust: {:.4f}\n".format(step, train_acc, train_loss, train_acc_r, train_loss_r))
                    f.close()
                ##
                if not os.path.exists(os.path.join(store_dir, "checkpoints")):
                    os.makedirs(os.path.join(store_dir, "checkpoints"))

            #### Validation ####
            if (step + 1) % args.test_period_print == 0:
                ### Saving training accs and loss
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                train_loss_r_list.append(train_loss_r)
                train_acc_r_list.append(train_acc_r)
                ###
                val_loss_avg, val_acc_avg, val_loss_r_avg, val_acc_r_avg = 0.0, 0.0, 0.0, 0.0

                if args.dataloader_mode == "few_shot":
                    for i, (x_spt, y_spt, x_qry, y_qry) in enumerate(validation_dl):
                        x_spt, y_spt, x_qry, y_qry = (
                            torch.stack(x_spt, dim=1).to(device),
                            torch.stack(y_spt, dim=1).to(device),
                            torch.stack(x_qry, dim=1).to(device),
                            torch.stack(y_qry, dim=1).to(device),
                        )
                        # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)
                        val_loss, val_acc, val_loss_r, val_acc_r = 0.0, 0.0, 0.0, 0.0
                        for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(
                                x_spt, y_spt, x_qry, y_qry):
                            if "_at" in args.model_name:
                                accs, loss, accs_r, loss_r = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                                val_loss += loss.detach().cpu().numpy()
                                val_acc += accs
                                val_loss_r += loss_r.detach().cpu().numpy()
                                val_acc_r += accs_r
                            else:
                                accs, loss = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                                val_loss += loss.detach().cpu().numpy()
                                val_acc += accs
                        val_loss /= x_spt.shape[0]
                        val_acc /= x_spt.shape[0]
                        val_loss_r /= x_spt.shape[0]
                        val_acc_r /= x_spt.shape[0]
                        val_loss_avg += val_loss
                        val_acc_avg += val_acc
                        val_loss_r_avg += val_loss_r
                        val_acc_r_avg += val_acc_r
                else:
                    for i, (x, y) in enumerate(validation_dl):
                        x, y = batch
                        x = x.to(device)
                        y = y.type(torch.LongTensor).to(device)
                        accs, loss, accs_r, loss_r = model.test(x, y)
                    val_loss_avg += loss
                    val_acc_avg += accs
                    val_loss_r_avg += loss_r
                    val_acc_r_avg += accs_r

                val_loss_avg /= (i + 1)
                val_acc_avg /= (i + 1)
                val_loss_r_avg /= (i + 1)
                val_acc_r_avg /= (i + 1)
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
                print("Epoch: {} Testing Acc: {:.4f}, Testing Loss: {:.4f}, Testing Acc Robust: {:.4f}, Testing Loss Robust: {:.4f} \n".format(step, val_acc_avg, val_loss_avg, val_acc_r_avg, val_loss_r_avg)
                      )
                with open(f"{store_dir}/results.txt", "a") as f:
                    f.writelines("Epoch: {} Testing Acc: {:.4f}, Testing Loss: {:.4f}, Testing Acc Robust: {:.4f}, Testing Loss Robust: {:.4f} \n".format(step, val_acc_avg, val_loss_avg, val_acc_r_avg, val_loss_r_avg))
                    f.close()
            step += 1
    else:
        #### Testing ####
        try:
            model.load_state_dict(torch.load(args.weight))  ####################### load the best weight ##########################
        except:
            params = torch.load(args.weight)
            model.load_state_dict(torch.load(params['weight']))
        if args.adv_attack == "LinfPGD":
            from attack import PGD
            at = PGD(eps=args.adv_attack_eps, sigma=args.adv_attack_alpha, nb_iter=args.adv_attack_iters)
        else:
            at = None
        test_loss_list, test_acc_list, test_loss_r_list, test_acc_r_list = [], [], [], []
        test_loss, test_acc, test_loss_r, test_acc_r = 0.0, 0.0, 0.0, 0.0
        test_loss_avg, test_acc_avg, test_loss_r_avg, test_acc_r_avg = 0.0, 0.0, 0.0, 0.0
        if args.dataloader_mode == "few_shot":
            for i, (x_spt, y_spt, x_qry, y_qry) in enumerate(test_dl):
                x_spt, y_spt, x_qry, y_qry = (
                    torch.stack(x_spt, dim=1).to(device),
                    torch.stack(y_spt, dim=1).to(device),
                    torch.stack(x_qry, dim=1).to(device),
                    torch.stack(y_qry, dim=1).to(device),
                )
                test_loss, test_acc, test_loss_r, test_acc_r = 0.0, 0.0, 0.0, 0.0
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(
                        x_spt, y_spt, x_qry, y_qry):
                    ###
                    if args.adv_attack == "LinfPGD":
                        # x_qry_one = at.attack(model.net, model.net.parameters(), x_qry_one, y_qry_one)
                        if args.model_name == "protonet_at" or args.model_name == "generic_protonet":
                            x_qry_one = at.attack(model.net, model.net.parameters(), x_qry_one, y_qry_one,
                                                  extra_params=dict(x_spt=x_spt_one, y_spt=y_spt_one))
                        else:
                            x_qry_one = at.attack(model.net, model.net.parameters(), x_qry_one, y_qry_one)
                    ###
                    if "_at" in args.model_name:
                        accs, loss, accs_r, loss_r = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        test_loss += loss.detach().cpu().numpy()
                        test_acc += accs
                        test_loss_r += loss_r.detach().cpu().numpy()
                        test_acc_r += accs_r
                    else:
                        accs, loss = model.test(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        test_loss += loss.detach().cpu().numpy()
                        test_acc += accs
                test_loss /= x_spt.shape[0]
                test_acc /= x_spt.shape[0]
                test_loss_r /= x_spt.shape[0]
                test_acc_r /= x_spt.shape[0]
                test_loss_avg += test_loss
                test_acc_avg += test_acc
                test_loss_r_avg += test_loss_r
                test_acc_r_avg += test_acc_r
        else:
            for i, (x, y) in enumerate(test_dl):
                x = x.to(device)
                y = y.type(torch.LongTensor).to(device)
                ###
                if args.adv_attack == "LinfPGD":
                    x = at.attack(model.net, model.net.parameters(), x, y)
                ###
                accs, loss, accs_r, loss_r = model.test(x, y)
                test_loss_avg += loss
                test_acc_avg += accs
                test_loss_r_avg += loss_r
                test_acc_r_avg += accs_r

            test_loss_avg /= (i + 1)
            test_acc_avg /= (i + 1)
            test_loss_r_avg /= (i + 1)
            test_acc_r_avg /= (i + 1)
            test_loss_list.append(test_loss_avg)
            test_acc_list.append(test_acc_avg)
            test_loss_r_list.append(test_loss_r_avg)
            test_acc_r_list.append(test_acc_avg)

            print(f"Epoch: {i}, test acc: {test_acc_avg} test loss {test_loss_avg}")
            with open(f"{os.path.dirname(args.weight)}/results_test.txt", 'a') as f:
                f.writelines(f"Epoch: {i}, test acc: {test_acc_avg} test loss {test_loss_avg} \n")
        total_acc_avg = sum(test_acc_list) / len(test_acc_list)
        total_loss_avg = sum(test_loss_list) / len(test_loss_list)
        print(f"Avg test acc: {total_acc_avg}  avg test loss {total_loss_avg}")
        with open(f"{os.path.dirname(args.weight)}/results_test.txt", 'a') as f:
            f.writelines(f"Test acc avg: {total_acc_avg} \n")
            f.writelines(f"Test loss avg: {total_loss_avg} \n")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    ### Model
    # Task 1: "generic_metanet" (1) "metanet_maml_at" (2) "generic_protonet" (3) "protonet_at" (4)
    # Task 2: "nonfs_classification" (5) "nonfs_classification_at" (6)
    # Task 2: "metanet_maml_at_gru" (7)  "protonet_at_gru" (8)
    argparser.add_argument('--model_name', type=str, help='The model name', default="nonfs_classification")
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1000)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)  # 0.001
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)  # 0.4
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--hidden_channel', type=int, help='hidden_channel', default=128)  # 64 128
    argparser.add_argument('--emb_channel', type=int, help='emb_channel', default=64)  # 64
    argparser.add_argument('--num_layers', type=int, help='The number of layers', default=3)
    argparser.add_argument('--weight', type=str, help='The learning phase', default=None)
    argparser.add_argument('--store_dir', type=str, help='The learning phase', default=None)

    ### Data
    argparser.add_argument('--mode', type=str, help='The learning phase', default="train")  # train test
    argparser.add_argument('--data_name', type=str, help='The data configuration', default="cifar10")  # omniglot cifar10
    argparser.add_argument('--img_shape', type=tuple, help='The image shape', default=(3, 32, 32))  # omniglot: (1, 28, 28)  cifar10: (3, 32, 32)
    argparser.add_argument('--dataloader_mode', type=str, help='dataloader mode', default="non_few_shot")  # few_show, non_few_shot

    ### Learning phases
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)  # 1000 5000
    argparser.add_argument('--n_way', type=int, help='n way', default=10)  # omniglot:5, cifar10: 10
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--train_period_print', type=int, help='train_period_print', default=1)
    argparser.add_argument('--test_period_print', type=int, help='test_period_print', default=5)
    argparser.add_argument('--test_size', type=int, help='test_size', default=5)
    argparser.add_argument('--weight_robust', type=float, help='weight_robust', default=1.0)  # 0.1 0.3 0.5 0.7 0.9 1.0 1.1 1.3 1.5 1.7 1.9

    ## Adversarial attack
    argparser.add_argument('--adv_attack_type', type=str, default="white_box", help="The adversarial attack type")  # white_box, black_box
    argparser.add_argument('--adv_attack', type=str, default="LinfPGD", help="The adversarial attack")  ### None LinfPGD FGSM
    argparser.add_argument('--adv_attack_eps', type=float, default=32 / 255, help="The adversarial attack pertuabation level value")  # 8/255 16/255 32/255 64/255 128/255 1
    argparser.add_argument('--adv_attack_alpha', type=float, default=4 / 255, help="The adversarial attack step size value")  # 4/255 16/255 32/255 64/255 128/255 1
    argparser.add_argument('--adv_attack_iters', type=float, default=7, help="The adversarial attack number of iterations value")  # 7 9 11 13 17 19
    argparser.add_argument('--adv_defense', type=str, default="AT", help="The adversarial defense")  ### None AT  TRADES
    ## Version control
    # argparser.add_argument('--leanrer_version', type=int, default=2, help="leanrer_version") # 1, 2
    args = argparser.parse_args()

    main(args)
