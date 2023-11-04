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


def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)
    return adv_label
    

def one_got_generator(input, num_classes, avg=True):
    one_hot = torch.zeros(input.shape[0], num_classes, device=input.device)
    if avg:
        one_hot += 1/num_classes
    else:
        one_hot.scatter_(1, B.view(-1, 1), 1)
    return one_hot
    
    
def train(model, train_loader, val_loader, optimizer, criterion, device, epoch=100, weight_dir="weights/checkpoints", store_dir="weights/checkpoints", num_classes=11, pretrain=False, attack=None, data_attack="all",
          lam_sep=1.0, lam_rec=1.0, eps_update=0, eps_delta=8/255, adv_threshold=0.5, period_adv_eps_update=None, percent_adv_eps_update=None, eval_adv=None, adv_tune=None):
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
    criterion = criterion
    model = model.to(device)
    # train the model for 100 epochs
    for epoch in range(epoch):
        print(f'Start epoch {epoch}')
        # initialize total loss
        total_loss = 0
        avg_loss = 0
        #####
        adv_cls_losses = 0 ##############
        sep_losses = 0 #################
        rec_losses = 0 ##############
        #adv_correct = 0
        #####
        if epoch != 0 and period_adv_eps_update is not None and (epoch+1) % period_adv_eps_update == 0:
            print("Update adv threshold value!!!")
            adv_threshold = (1 + percent_adv_eps_update) * adv_threshold
            adv_threshold = np.min([adv_threshold, 0.999])
        # iterate over triplets of data
        for batch_idx, (images, labels) in enumerate(train_loader):
            #set to val mode - change the model emp to same to model triplet
            images = images.to(device)
            labels = labels.to(device)
            if attack is not None:
                model.eval()
                if data_attack == "all":
                    images_adv = attack.perturb(images.float(), labels)
                    #images_adv = attack.perturb(images.float(), labels, attack.eps + (k+1)*eps_delta, attack.alpha + (k+1)*2/255)
                    #images_adv = images
                elif data_attack == "1vs1":
                    if adv_tune is not None:
                        images_adv = attack.perturb(images.float(), labels, attack.eps, attack.alpha, tune=True, val=0.0, val_mode=False)
                        #images_adv = attack.perturb(images.float(), labels, tune=True)
                    else:
                        #images_adv = attack.perturb(images.float(), labels, attack.eps + (k+1)*eps_delta, attack.alpha + (k+1)*2/255)
                        images_adv = attack.perturb(images.float(), labels)
                    #images = torch.cat([images, images_adv])
                    #labels = torch.cat([labels, labels])
            # start training
            model.train()
            optimizer.zero_grad()
            training_time = time.time()
            #####
            
            #####
            # Output
            # output = model(images.float())
            ############################ FSR output ###########################################
            adv_outputs, adv_r_outputs, adv_nr_outputs = model(images_adv.float(), is_train=True)
            ######################################################################
            if eps_update > 0:
                eps = attack.eps
                output = model(images.float())
                res = predict(output, labels)
                res_adv = predict(adv_outputs, labels)
                for k in range(eps_update):    
                    if res_adv >= res or res_adv >= int(images.size(0)) * adv_threshold:
                        images_adv = attack.perturb(images.float(), labels, attack.eps + (k+1)*eps_delta, attack.alpha + (k+1)*2/255, tune=True, val=0.0, val_mode=False)
                        adv_outputs, adv_r_outputs, adv_nr_outputs = model(images_adv.float(), is_train=True)
                    else: 
                        break
                    res_adv = predict(adv_outputs, labels)
            else:
                output = model(images.float())
                    
            ###
            ############################# Gating ################################################
            #adv_outputs, adv_r_outputs, adv_nr_outputs, adv_rec_outputs = model(images_adv.float(), is_train=True)
            #adv_labels = get_pred(adv_outputs, labels) ################

            adv_cls_loss = criterion(adv_outputs, labels) ##############
            
            r_loss = torch.tensor(0.).to(device) ##################
            r_out_sum = torch.zeros_like(adv_outputs).to(device)
            if not len(adv_r_outputs) == 0:
                for r_out in adv_r_outputs:
                    r_out_sum += r_out
                    r_loss += lam_sep * criterion(r_out, labels)
                r_loss /= len(adv_r_outputs)
                r_sum_loss = criterion(r_out_sum, labels) ##############################

            nr_loss = torch.tensor(0.).to(device) ##################
            if not len(adv_nr_outputs) == 0:
                for nr_out in adv_nr_outputs:
                    adv_labels = get_pred(nr_out, labels) ############## test ############################
                    nr_loss += lam_sep * criterion(nr_out, adv_labels)
                nr_loss /= len(adv_nr_outputs)
            sep_loss = r_loss + r_sum_loss + nr_loss ###############

            #rec_loss = torch.tensor(0.).to(device) ###############
            #if not len(adv_rec_outputs) == 0:
            #    for rec_out in adv_rec_outputs:
            #        rec_loss += lam_rec * criterion(rec_out, labels)
            #    rec_loss /= len(adv_rec_outputs)

            if data_attack == "1vs1":
                cls_loss = criterion(output, labels)
                loss = adv_cls_loss + cls_loss + sep_loss
            else:
                loss = adv_cls_loss + sep_loss ####################
            ############################# FSR ################################################
            #loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # calculate the average loss
            total_loss += loss
            # print the training progress after each epoch
            if eps_update > 0:
                print('Epoch: {} Batch: {} Loss: {:.4f} Res_adv: {} Res: {} Adv_update_iters: {}, Adv_threshold: {}'.format(epoch, batch_idx ,loss, res_adv, res, k, adv_threshold))
            else:
                print('Epoch: {} Batch: {} Loss: {:.4f}'.format(epoch, batch_idx ,loss))
        avg_loss = total_loss / (batch_idx + 1)
        total_result = 0
        total_sample = 0
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            if eval_adv is not None:
                images_adv_list = []
                labels_list = []
                for eval_eps in eval_adv:
                    images_adv = attack.perturb(images.float(), labels, eval_eps, new_alpha=None, tune=False, val=0.0, val_mode=True)
                    images_adv_list.append(images_adv)
                    labels_list.append(labels)
                images_adv_list.append(images)
                labels_list.append(labels)    
                images = torch.cat(images_adv_list)
                labels = torch.cat(labels_list)
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