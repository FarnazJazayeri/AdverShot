###### https://github.com/yaodongyu/TRADES/tree/master
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np
import os

def update_learning_rate(optimizer, decay_rate=0.999, lowest=0.00005):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr


def predict_logictics(outputs, labels, threshold=0.5):
    predictions = []
    _, predicted_classes = torch.max(outputs, dim=1)
    predicted_probabilities, _ = torch.max(outputs, dim=1)
    predicted_classes[predicted_probabilities < threshold] = 33
    predictions.append(predicted_classes.cpu().numpy())
    predictions = np.concatenate(predictions)
    ###
    label_list = []
    label_np = labels.cpu().numpy()
    for i in range(label_np.shape[0]):
        if max(label_np[i,:]) == 0:
            index = 33
            label_list.append(index)
        else:
            index = np.where(label_np[i,:] == 1)
            label_list.append(index[0][0])
    label_index = np.array(label_list)
    #####
    result = (label_index  == predictions).astype(int)
    result = result.sum()
    return result


def predict(outputs, labels, threshold=None):
    predictions = []
    _, predicted_classes = torch.max(outputs, dim=1)
    predicted_probabilities, _ = torch.max(outputs, dim=1)
    predictions.append(predicted_classes.cpu().numpy())
    predictions = np.concatenate(predictions)
    ###
    label_list = []
    label_np = labels.cpu().numpy()
    result = (label_np == predictions).astype(int)
    result = result.sum()
    return result
    
def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


'''
def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss
'''
    
def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)

    return adv_label
    
        
def train(model, train_loader, val_loader, optimizer, criterion, device, epoch=100, weight_dir="weights/checkpoints", store_dir="weights/checkpoints", num_classes=22, pretrain=False, attack=None, data_attack="all", beta=1.0, lam_sep=1.0, lam_rec=1.0):
    # set the model to train mode
    if pretrain:
        print('Load the last checkpoint...')
        model.load_state_dict(torch.load(weight_dir))
    else:
        print('Start training without reload.')
        pass
    #set the device
    if device == 'cpu':
        device = torch.device('cpu')
    elif device == 'gpu':
        device = torch.device('cuda:0')
    best_loss = 0
    #bce_loss = nn.BCELoss()
    criterion = criterion
    model = model.to(device)
    kl = nn.KLDivLoss(reduction='none')
    # train the model for 100 epochs
    for epoch in range(epoch):
        print(f'Start epoch {epoch}')
        # initialize total loss
        total_loss = 0
        avg_loss = 0
        start_time = time.time()
        # iterate over triplets of data
        for batch_idx, (images, labels) in enumerate(train_loader):
            #set to val mode - change the model emp to same to model triplet
            # model.swin.eval()
            images = images.to(device)
            labels = labels.to(device)
            if attack is not None:
                if data_attack == "all":
                    images_adv = attack.perturb(images.float(), labels)
                elif data_attack == "1vs1":
                    images_adv = attack.perturb(images.float(), labels)
                    #images_cat = torch.cat([images, images_adv])
                    #labels = torch.cat([labels, labels])
            # start training
            model.train()
            optimizer.zero_grad()
            training_time = time.time()
            # Output
            #output = model(images.float())
            ############################# FSR ################################################
            adv_outputs, adv_r_outputs, adv_nr_outputs, adv_rec_outputs = model(images_adv.float(), is_train=True)
            adv_labels = get_pred(adv_outputs, labels) ################

            # adv_cls_loss = criterion(adv_outputs, labels) ##############
            
            r_loss = torch.tensor(0.).to(device) ##################
            if not len(adv_r_outputs) == 0:
                for r_out in adv_r_outputs:
                    r_loss += lam_sep * criterion(r_out, labels)
                r_loss /= len(adv_r_outputs)

            nr_loss = torch.tensor(0.).to(device) ##################
            if not len(adv_nr_outputs) == 0:
                for nr_out in adv_nr_outputs:
                    nr_loss += lam_sep * criterion(nr_out, adv_labels)
                nr_loss /= len(adv_nr_outputs)
            sep_loss = r_loss + nr_loss ###############

            rec_loss = torch.tensor(0.).to(device) ###############
            if not len(adv_rec_outputs) == 0:
                for rec_out in adv_rec_outputs:
                    rec_loss += lam_rec * criterion(rec_out, labels)
                rec_loss /= len(adv_rec_outputs)
            
            # MART
            batch_size = images.shape[0]
            logits = model(images.float())
            logits_adv = adv_outputs
            adv_probs = F.softmax(logits_adv, dim=1)
            tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
            new_y = torch.where(tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1])
            adv_cls_loss = F.cross_entropy(logits_adv, labels) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
            nat_probs = F.softmax(logits, dim=1)
            true_probs = torch.gather(nat_probs, 1, (labels.unsqueeze(1)).long()).squeeze()
            loss_robust = (1.0 / batch_size) * torch.sum(
                torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
                
            loss = adv_cls_loss + sep_loss + rec_loss + loss_robust  ####################
            ############################# FSR ################################################
            
            loss.backward()
            optimizer.step()
            # calculate the average loss
            total_loss += loss
            # print the training progress after each epoch
            total_time = time.time() - start_time
            print('Epoch: {} Batch: {} Loss: {:.4f}'.format(epoch, batch_idx ,loss))
        avg_loss = total_loss / (batch_idx + 1)
        total_result = 0
        total_sample = 0
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            if attack is not None:
                images_adv = attack.perturb(images.float(), labels)
                images = torch.cat([images, images_adv])
                labels = torch.cat([labels, labels])
            with torch.no_grad():
                model.eval()
                output = model(images.float())
                results = predict(output, labels, threshold=0.5)
                total_result += results
                total_sample += labels.size()[0]
        val_percent = total_result/total_sample
        print(f"Validation results of Epoch {epoch}: {total_result}/{total_sample} | {val_percent} %")
        #Save model
        if not os.path.exists(os.path.join(store_dir, 'checkpoints')):
            os.makedirs(os.path.join(store_dir, 'checkpoints'))
        torch.save(model.state_dict(), f'{store_dir}/checkpoints/last.pt')
        # deep copy the model
        if epoch == 0:
            best_percent = val_percent
            #save best weight
            print('Save intitial weight')
            torch.save(model.state_dict(), f'{store_dir}/checkpoints/init.pt')
        elif best_percent < val_percent:
            if val_percent < 1:
                best_percent = val_percent
                #save best weight
                print(f'Save best weight: {best_percent}')
                torch.save(model.state_dict(), f'{store_dir}/checkpoints/best_new.pt')
        else:
            update_learning_rate(optimizer)

        # save weight each epoch
        #print('Save each epoch - batch weight')
        #torch.save(model.state_dict(), f'{weight_dir}/epoch{int(epoch)}_batch{batch_idx}.pt')
        torch.cuda.empty_cache()
        print('Epoch: {} Avg Loss: {:.4f}, Val percent: {}'.format(epoch, avg_loss, val_percent))
        with open(f'{store_dir}/results.txt', 'a') as f:
            f.writelines('Epoch: {} Avg Loss: {:.4f}, Val percent: {} \n'.format(epoch, avg_loss, val_percent))
            f.close()
