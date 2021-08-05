# train.py
#!/usr/bin/env	python3

""" train network using pytorch
author staylor
"""

import os
import argparse
import time

import torch
from torch.functional import norm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models


from torch.utils.data import DataLoader, dataloader
# from torch.utils.tensorboard import SummaryWriter

from dataset.sampled_flame_dataset import SampledFlameDataset

from models.simsiam import SimSiam  
from models.barlowtwins import Transform, adjust_learning_rate

def train(epoch):

    epoch_loss = 0.0
    start = time.time()
    net.train()
    print('Start training')
    for batch_index, image in enumerate(training_loader):

        # randomly manipulate
        y1, y2 = trans.__call__(image)
        if args.gpu:
            y1 = y1.cuda()
            y2 = y2.cuda()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        adjust_learning_rate(args.epochs, args.b, optimizer, training_loader, n_iter)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            p_left, z_right, p_right, z_left = net(y1, y2)
            loss = -(loss_function(p_left, z_right).mean() + loss_function(p_right, z_left).mean()) * 0.5


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        # loss.backward()
        # optimizer.step()


        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b,
            total_samples=len(training_loader.dataset)
        ))

        #update training loss for each iteration
        # writer.add_scalar('Train/loss', loss.item(), n_iter)

        # if epoch <= args.warm:
        #     warmup_scheduler.step()

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

    return epoch_loss / len(training_loader)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-checkpoint',  default='checkpoints', help='checkpoint directory')
    parser.add_argument('-logdir',  default='tensorboard', help='checkpoint directory')
    parser.add_argument('-b', type=int, default=19, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    setattr(args, "projector", '8192-8192-8192')
    setattr(args, "batch_size", args.b)
    setattr(args, "lambd", 0.0051)
    setattr(args, "weight_decay", 1e-6)
    
    
    net = SimSiam()
    if args.gpu:
        net = net.cuda()

    param_weights = []
    param_biases = []
    for param in net.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]

    dataset = SampledFlameDataset(dataset_len = 180, device='cuda')
    print(f'{len(dataset):d} training samples')
    
    training_loader = DataLoader(dataset, shuffle=True, num_workers=0, batch_size=args.b)

    loss_function = nn.CosineSimilarity()
    optimizer = optim.Adam(parameters, lr=args.lr)

    checkpoint_path = args.checkpoint

    #use tensorboard
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

   
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, 'resnet50-{epoch}-{type}.pth')

    best_loss = 1e10

    scaler = torch.cuda.amp.GradScaler()
    trans = Transform()
    for epoch in range(1, args.epochs+1):


        # train(epoch, scaler, trans)
        # acc = eval_training(trans, epoch)        
        test_loss = train(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > 5 and test_loss < best_loss:
            weights_path = checkpoint_path.format(net='barlow', epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_loss = test_loss
            continue

        if not epoch % 10:
            weights_path = checkpoint_path.format(net='barlow', epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    # writer.close()
