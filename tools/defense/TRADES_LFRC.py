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
    
def train(model, train_loader, val_loader, optimizer, criterion, device, epoch=100, weight_dir="weights/checkpoints", store_dir="weights/checkpoints", num_classes=22, pretrain=False, attack=None, data_attack="all", beta=1.0, feature_indexes=[0, 1, 2, 3]):
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
        device = torch.device('cuda:1')
    best_loss = 0
    #bce_loss = nn.BCELoss()
    criterion = criterion
    model = model.to(device)
    criterion_kl = nn.KLDivLoss(size_average=False)
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
                model.eval()
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
            ###### TRADES loss ###############################
            batch_size = images.shape[0]
            
            ############################ LFRC ##################################
            # Output
            if feature_indexes is not None:
                output, features = model(images.float(), feature_return=feature_indexes)
                output_adv, features_adv = model(images_adv.float(), feature_return=feature_indexes)
                
                loss = criterion(output_adv, labels)
                loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(output_adv, dim=1),
                                                            F.softmax(output, dim=1))
                for index in feature_indexes:
                    normed_clean = F.normalize(features[index], dim=-1) 
                    matrix_clean = torch.mm(normed_clean, normed_clean.t()) ##########################################
        
                    normed_feature = F.normalize(features_adv[index], dim=-1)
                    matrix_adv = torch.mm(normed_feature, normed_feature.t()) #######################################
        
                    diff = torch.exp(torch.abs(matrix_adv - matrix_clean)) #####################################
                    loss_lfrc = 0.1*torch.mean(diff) ##################################### 1 100
                    loss += loss_lfrc
                loss += loss_robust 
                loss.backward()
            else:
                output = model(images_adv.float())
                loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(images_adv.float()), dim=1),
                                                            F.softmax(output, dim=1))
                loss = criterion(output, labels)
                loss += loss_robust
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
