# train.py
#!/usr/bin/env	python3

""" train network using pytorch
author staylor
"""

import os
import argparse
import time
import math

import torch
# from torch.functional import norm
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.sampled_flame_dataset import SampledFlameDataset

from models.resnet_encoder_decoder import ResNetEncDec, SSIM  
from torchvision import transforms


def train(epoch):

    epoch_loss = 0.0
    start = time.time()
    net.train()
    print('Start training')
    for batch_index, (_, img_norm, _, img_aug_norm) in enumerate(training_loader):

        if args.gpu:
            img_norm = img_norm.cuda()
            img_aug_norm = img_aug_norm.cuda()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        adjust_learning_rate(args.epochs, args.lr, optimizer, training_loader, n_iter)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
    
            _, orig_recon_img = net(img_norm)
            _, aug_recon_img = net(img_aug_norm)

            # latent_loss = -(loss_function(l_latent, r_latent.detach()).mean() + loss_function(r_latent, l_latent.detach()).mean()) * 0.5
            # latent_loss = loss_function(l_latent, r_latent)
            # photo_loss = (photo_loss_function(l_img, img_norm) + photo_loss_function(r_img, img_aug_norm)) * 0.5
            

            # compute errors between both reconstructed images and the ORIGINAL image
            l1_loss_l = torch.abs(img_norm - orig_recon_img).mean()
            ssim_loss_l = photo_loss_function(orig_recon_img, img_norm).mean()
            loss_l = 0.85 * ssim_loss_l + 0.15 * l1_loss_l

            l1_loss_r = torch.abs(img_norm - aug_recon_img).mean()
            ssim_loss_r = photo_loss_function(aug_recon_img, img_norm).mean()
            loss_r = 0.85 * ssim_loss_r + 0.15 * l1_loss_r

            photo_loss = (loss_l + loss_r)*0.5

            # loss = latent_loss + photo_loss
            loss =  photo_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()



        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        # print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLatent Loss: {:0.4f}\tPhoto Loss: {:0.4f}\tLR: {:0.6f}'.format(
        #     loss.item(),
        #     latent_loss.item(),
        #     photo_loss.item(),
        #     optimizer.param_groups[0]['lr'],
        #     epoch=epoch,
        #     trained_samples=batch_index * args.b,
        #     total_samples=len(training_loader.dataset)
        # ))


        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b,
            total_samples=len(training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        # writer.add_scalar('Train/latentloss', latent_loss.item(), n_iter)
        # writer.add_scalar('Train/photoloss', photo_loss.item(), n_iter)
        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], n_iter)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

    return epoch_loss / len(training_loader)


def visualise_result(reconstruction_path, epoch):
    img, img_norm, img_aug, img_aug_norm = dataset[0]
    _, reconstructed_img = net(img_norm[None,...].cuda())
    reconstructed_img = reconstructed_img.detach().cpu()
    reconstructed_img.mul_(torch.tensor([0.229, 0.224, 0.225])[:,None,None].repeat(1,384,384)).add_(torch.tensor([0.485, 0.456, 0.406])[:,None,None].repeat(1,384,384))
    reconstructed_img_vis = transforms.ToPILImage()(reconstructed_img[0])
    img_vis = transforms.ToPILImage()(img)
    reconstructed_img_vis.save("{:s}/{:03d}_reconstructed.jpg".format(reconstruction_path,epoch))
    img_vis.save("{:s}/{:03d}_original.jpg".format(reconstruction_path,epoch))

    _, reconstructed_img = net(img_aug_norm[None,...].cuda())
    reconstructed_img = reconstructed_img.detach().cpu()
    reconstructed_img.mul_(torch.tensor([0.229, 0.224, 0.225])[:,None,None].repeat(1,384,384)).add_(torch.tensor([0.485, 0.456, 0.406])[:,None,None].repeat(1,384,384))
    reconstructed_img_vis = transforms.ToPILImage()(reconstructed_img[0])
    img_vis = transforms.ToPILImage()(img_aug)
    reconstructed_img_vis.save("{:s}/{:03d}_aug_reconstructed.jpg".format(reconstruction_path,epoch))
    img_vis.save("{:s}/{:03d}_aug.jpg".format(reconstruction_path,epoch))


# def adjust_learning_rate(epochs, base_lr, optimizer, loader, step, learning_rate_weights=0.2, learning_rate_biases= 0.0048):
def adjust_learning_rate(epochs, base_lr, optimizer, loader, step, learning_rate_weights=1.0, learning_rate_biases= 1.0):
    max_steps = epochs * len(loader)
    warmup_steps = 10 * len(loader)
    # base_lr = batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * learning_rate_biases

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-checkpoint',  default='checkpoints', help='checkpoint directory')
    parser.add_argument('-reconstruction',  default='reconstruction', help='checkpoint directory')
    parser.add_argument('-logdir',  default='tensorboard', help='checkpoint directory')
    parser.add_argument('-b', type=int, default=19, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    setattr(args, "batch_size", args.b)

    
    net = ResNetEncDec()

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

    dataset = SampledFlameDataset(dataset_len = 2000, device='cuda')
    print(f'{len(dataset):d} training samples')
    
    training_loader = DataLoader(dataset, shuffle=True, num_workers=0, batch_size=args.b)

    # loss_function = nn.CosineSimilarity()
    # photo_loss_function = SSIMLoss()
    # loss_function = nn.MSELoss()
    photo_loss_function = SSIM()
    optimizer = optim.Adam(parameters, lr=args.lr)

    checkpoint_path = args.checkpoint
    reconstruction_path = args.reconstruction

    #use tensorboard
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    
    writer = SummaryWriter(args.logdir)


   
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    if not os.path.exists(reconstruction_path):
        os.makedirs(reconstruction_path)

    best_loss = 1e10

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, args.epochs+1):

        test_loss = train(epoch)

        net.eval()

        # check an image
        visualise_result(reconstruction_path, epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > 5 and test_loss < best_loss:
            weights_path = checkpoint_path.format(net='encdec', epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_loss = test_loss
            continue

        if not epoch % 10:
            weights_path = checkpoint_path.format(net='encdec', epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()

