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


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.face_image_dataset import FaceImageDataset

from models.barlowtwins import BarlowTwins, LARS, exclude_bias_and_norm, Transform, adjust_learning_rate


def train(epoch):


    start = time.time()
    net.train()
    print('Start training')
    for batch_index, (image, label) in enumerate(training_loader):

        # randomly manipulate
        y1, y2 = trans.__call__(image)
        if args.gpu:
            y1 = y1.cuda()
            y2 = y2.cuda()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        adjust_learning_rate(args.epochs, args.b, optimizer, training_loader, n_iter)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            z1, z2 = net(y1, y2)
            loss = loss_function(z1, z2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        # if epoch <= args.warm:
        #     warmup_scheduler.step()

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error

    for (image, label) in test_loader:

        # randomly manipulate
        y1, y2 = trans.__call__(image)
        if args.gpu:
            y1 = y1.cuda()
            y2 = y2.cuda()


        with torch.cuda.amp.autocast():
            z1, z2 = net(y1, y2)
            loss = loss_function(z1, z2)
        # scaler.scale(loss)

        test_loss += loss.item()
        
    finish = time.time()
    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f},  Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)

    return test_loss / len(test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-checkpoint',  default='checkpoints', help='checkpoint directory')
    parser.add_argument('-logdir',  default='tensorboard', help='checkpoint directory')
    parser.add_argument('-b', type=int, default=10, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    setattr(args, "projector", '8192-8192-8192')
    setattr(args, "batch_size", args.b)
    setattr(args, "lambd", 0.0051)
    setattr(args, "weight_decay", 1e-6)
    
    
    # net = resnet18()
    # net = ResNet18_Pretrained()
    net = BarlowTwins(args)
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

    dataset = FaceImageDataset(['/mnt/hdd/data1/modality/shuffled/processed_data/train/','/mnt/hdd/data1/modality/shuffled/processed_data/train_synthetic'], norm_images=False)
    print(f'{len(dataset):d} training samples')
    
    training_loader = DataLoader(dataset, shuffle=True, num_workers=1, batch_size=args.b)



    testdataset = FaceImageDataset('/mnt/hdd/data1/modality/shuffled/processed_data/test/', norm_images=False)
    test_loader = DataLoader(testdataset, shuffle=False, num_workers=1, batch_size=min(len(testdataset),args.b))

    loss_function = net.compute_loss
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=exclude_bias_and_norm,
                     lars_adaptation_filter=exclude_bias_and_norm)

    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    # iter_per_epoch = len(cifar100_training_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # if args.resume:
    #     recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
    #     if not recent_folder:
    #         raise Exception('no recent folder were found')

    #     checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    # else:
    # checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    checkpoint_path = args.checkpoint

    #use tensorboard
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    # writer = SummaryWriter(log_dir=os.path.join(
            # settings.LOG_DIR, args.net, settings.TIME_NOW))
    writer = SummaryWriter(log_dir=args.logdir)
    # input_tensor = torch.Tensor(1, 3, 32, 32)
    # if args.gpu:
    #     input_tensor = input_tensor.cuda()
    # writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, 'barlow-{epoch}-{type}.pth')

    best_loss = 1e10
    # if args.resume:
    #     best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    #     if best_weights:
    #         weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
    #         print('found best acc weights file:{}'.format(weights_path))
    #         print('load best training file to test acc...')
    #         net.load_state_dict(torch.load(weights_path))
    #         best_acc = eval_training(tb=False)
    #         print('best acc is {:0.2f}'.format(best_acc))

    #     recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    #     if not recent_weights_file:
    #         raise Exception('no recent weights file were found')
    #     weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
    #     print('loading weights file {} to resume training.....'.format(weights_path))
    #     net.load_state_dict(torch.load(weights_path))

    #     resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    scaler = torch.cuda.amp.GradScaler()
    trans = Transform()
    for epoch in range(1, args.epochs+1):


        # train(epoch, scaler, trans)
        # acc = eval_training(trans, epoch)        
        train(epoch)
        test_loss = eval_training(epoch)

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

    writer.close()
