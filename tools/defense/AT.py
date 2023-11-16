import torch
import torch.nn as nn
import time
import os
import numpy as np


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


def train(model, train_loader, val_loader, optimizer, criterion, device, epoch=100, weight_dir="weights/checkpoints", store_dir="weights/checkpoints", num_classes=22, pretrain=False,
          attack=None, data_attack="all", eps_update=0, eps_delta=8/255, adv_threshold=0.5, period_adv_eps_update=None, percent_adv_eps_update=None, eval_adv=None, adv_tune=None):
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
    # train the model for 100 epochs
    val_acc = 0.0
    for epoch in range(epoch):
        print(f'Start epoch {epoch}')
        # initialize total loss
        total_loss = 0
        avg_loss = 0
        #start_time = time.time()
        # iterate over triplets of data
        if epoch != 0 and period_adv_eps_update is not None and (epoch+1) % period_adv_eps_update == 0:
            print("Update adv threshold value!!!")
            adv_threshold = (1 + percent_adv_eps_update) * adv_threshold
            adv_threshold = np.min([adv_threshold, 0.9])
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
                    if adv_tune is not None:
                        images_adv = attack.perturb(images.float(), labels, tune=True)
                    else:
                        images_adv = attack.perturb(images.float(), labels)
                    #images = torch.cat([images, images_adv])
                    #labels = torch.cat([labels, labels])
            # start training
            model.train()
            optimizer.zero_grad()
            #training_time = time.time()
            # Output
            ###
            output_adv = model(images_adv.float())
            if epoch != 0 and period_adv_eps_update is not None and (epoch+1) % period_adv_eps_update == 0:
                print("Update adv threshold value!!!")
                adv_threshold = (1 + percent_adv_eps_update) * adv_threshold
                adv_threshold = np.min([adv_threshold, 0.9])
            if eps_update > 0:
                eps = attack.eps
                output = model(images.float())
                res = predict(output, labels)
                res_adv = predict(output_adv, labels)
                for k in range(eps_update):    
                    if res_adv >= res or res_adv >= int(images.size(0)) * adv_threshold:
                        images_adv = attack.perturb(images.float(), labels, attack.eps + (k+1)*eps_delta)
                        output_adv = model(images_adv.float())
                    else: 
                        break
                    res_adv = predict(output_adv, labels)
                    
            ###
            loss = criterion(output_adv, labels)
            loss.backward()
            optimizer.step()
            # calculate the average loss
            total_loss += loss
            # print the training progress after each epoch
            if eps_update > 0:
                print('Epoch: {} Batch: {} Loss: {:.4f} Res_adv: {} Res: {} Adv_update_iters: {}'.format(epoch, batch_idx ,loss, res_adv, res, k))
            else:
                print('Epoch: {} Batch: {} Loss: {:.4f}'.format(epoch, batch_idx ,loss))
        avg_loss = total_loss / (batch_idx + 1)
        total_result = 0
        total_sample = 0
        for batch_idx, (images, labels) in enumerate(val_loader):
            #print(batch_idx)
            if batch_idx == 35:
                break
            images = images.to(device)
            labels = labels.to(device)
            if attack is not None:
                model.eval()
                if eval_adv is not None:
                    images_adv_list = []
                    labels_list = []
                    for eval_eps in eval_adv:
                        images_adv = attack.perturb(images.float(), labels, eval_eps)
                        images_adv_list.append(images_adv)
                        labels_list.append(labels)
                    images_adv_list.append(images)
                    labels_list.append(labels)    
                    images = torch.cat(images_adv_list)
                    labels = torch.cat(labels_list)
                else:
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
        attack.update_features(val=val_percent)
        print(f"Validation results of Epoch {epoch}: {total_result}/{total_sample} | {val_percent} %")
        #Save model
        if not os.path.exists(os.path.join(store_dir, 'checkpoints')):
            os.makedirs(os.path.join(store_dir, 'checkpoints'))
        # deep copy the model
        torch.save(model.state_dict(), f'{store_dir}/checkpoints/last.pt')
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