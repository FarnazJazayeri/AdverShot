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


def train(model, train_loader, val_loader, optimizer, criterion, device, epoch=100, weight_dir="weights/checkpoints", store_dir="weights/checkpoints", num_classes=22, pretrain=False, attack=None, data_attack="all"):
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
                    images = attack.perturb(images.float(), labels)
                elif data_attack == "1vs1":
                    images_adv = attack.perturb(images.float(), labels)
                    images = torch.cat([images, images_adv])
                    labels = torch.cat([labels, labels])
            # start training
            model.train()
            optimizer.zero_grad()
            training_time = time.time()
            # Output
            output = model(images.float())
            #loss = bce_loss(output[:,0], labels[:, 0])
            #for i in range(1,num_classes):
            #    loss += bce_loss(output[:,i], labels[:, i])
            loss = criterion(output, labels)
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
                model.eval()
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