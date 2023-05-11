import argparse
import os
import math
import time
import random
import warnings
import numpy as np
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model_AE import *
from dataset import Transform, Transform_noaug
from utils import AverageMeter

from PIL import Image


parser = argparse.ArgumentParser(description='Autoencoder Training')
# General variables
parser.add_argument('--data', metavar='DIR', default='/data/datasets/imagenette2/')
parser.add_argument('--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--save_dir', type=str, default='./experiments/testing/')
parser.add_argument('--workers', type=int, metavar='N', default=32)
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--checkpoint_freq', type=int, default=10)

# Training variables
parser.add_argument('--epochs', type=int, metavar='N', default=50)#100
parser.add_argument('--batch_size', type=int, metavar='N', default=128) # 1024
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, metavar='W', default=1e-6)
parser.add_argument('--stop_epoch', type=int, default=None)
parser.add_argument('--no_aug', action='store_true')

# SEED
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')

# GPU
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

def main():
    args = parser.parse_args()
    print(vars(args))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_dir_models = os.path.join(args.save_dir, 'models')
    if not os.path.exists(args.save_dir_models):
        os.makedirs(args.save_dir_models)

    # Set the seed
    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    main_worker(args)
        
        
def main_worker(args):
    start_time_all = time.time()

    print('\nUsing Single GPU training')
    print('Use GPU: {} for training'.format(args.gpu))

    print('\nLoading dataloader ...')
    transform_funct = Transform_noaug() if args.no_aug else Transform()
    train_dataset = datasets.ImageFolder(os.path.join(args.data,'train'), transform_funct)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True)
    val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224), 
                        transforms.ToTensor(),])
    val_dataset = datasets.ImageFolder(os.path.join(args.data,'val'), val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
                                               shuffle=False, num_workers=args.workers,
                                               pin_memory=True)
    
    print('\nLoading model, optimizer, and lr scheduler...')
    model = Autoencoder(args)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, 
                                weight_decay=args.weight_decay)
    
    print('\nStart training...')
    start_epoch=0
    all_losses = []
    for epoch in range(start_epoch, args.epochs):
        start_time_epoch = time.time()

        # train the network
        epoch_loss = train(
            train_loader,
            model,
            optimizer,
            epoch,
            args,
        )
        all_losses.append(epoch_loss)

        # validation and save reconstruccions
        epoch_val_loss = validate(val_loader, model, args)
        print("Epoch: [{}] | Train loss: {:.4f} | Val loss: {:.4f}".format(epoch, epoch_loss, epoch_val_loss))

        # save checkpoints and encoder
        save_checkpoints_encoder(model, optimizer, epoch, args)

         # save losses
        np.save(os.path.join(args.save_dir, "losses.npy"), np.array(all_losses))

        # print time per epoch
        end_time_epoch = time.time()
        print("Epoch time: {:.2f} minutes".format((end_time_epoch - start_time_epoch) / 60),
              "Training time: {:.2f} minutes".format((end_time_epoch - start_time_all) / 60))
    
        if (epoch+1) == args.stop_epoch:
            break

    end_time_all = time.time()
    print("Total training time: {:.2f} minutes".format((end_time_all - start_time_all) / 60))


def train(loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    criterion = nn.MSELoss().cuda(args.gpu)

    end = time.time()
    for it, (inputs, labels) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda(args.gpu, non_blocking=True)
        output = model(inputs)
        loss = criterion(output, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # misc
        losses.update(loss.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if it % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return losses.avg

def validate(val_loader, model, args):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    criterion = nn.MSELoss()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            recon = model(inputs)
            loss = criterion(recon, inputs)

            # misc
            losses.update(loss.item(), inputs.size(0))

            # save reconstructed images
            disp = torch.cat((inputs[:20], recon[:20]), axis=0)
            imgdisp(disp.cpu(), i, args)

    return losses.avg



class Autoencoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = ResNet18Enc(zero_init_residual=True)
        self.decoder = ResNet18Dec(zero_init_residual=True)

    def forward(self, x):
        # encoder
        r = self.encoder(x)
        # decoder
        x_hat = self.decoder(r)
        return x_hat
    
def save_checkpoints_encoder(model, optimizer, epoch, args):
    # save autoencoder
    save_dict = {"epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),}
    torch.save(save_dict, os.path.join(args.save_dir , "checkpoint.pth"))
    if (epoch+1) % args.checkpoint_freq == 0 or (epoch+1) == args.epochs or (epoch+1) == args.stop_epoch:
        shutil.copyfile(
            os.path.join(args.save_dir, "checkpoint.pth"),
            os.path.join(args.save_dir_models, "ckp_epoch" + str(epoch+1) + ".pth"))

    # save encoder
    save_dict = {"epoch": epoch + 1,
                 "state_dict": model.encoder.state_dict()}
    torch.save(save_dict, os.path.join(args.save_dir , str(args.arch) + ".pth"))
    if (epoch+1) % args.checkpoint_freq == 0 or (epoch+1) == args.epochs or (epoch+1) == args.stop_epoch:
        shutil.copyfile(
            os.path.join(args.save_dir, str(args.arch) + ".pth"),
            os.path.join(args.save_dir_models, str(args.arch) + "_epoch" + str(epoch+1) + ".pth"))
        
    return None

def imgdisp(images, iter, args):
    # images.shape[0]=40
    m = int(images.shape[0]/10)
    n = int(images.shape[0]/m)
    collage = Image.new("RGB", (224*n, 224*m))
    for i in range(0, m):
        for j in range(0, n):
            offset = 224 * j, 224 * i
            idx = i * n + j
            trans = transforms.ToPILImage()
            if idx < 10:
                img = trans(images[idx])
            elif idx > 9 and idx < 20:
                img = trans(images[idx+10])
            elif idx > 19 and idx < 30:
                img = trans(images[idx-10])
            elif idx > 29:
                img = trans(images[idx])
            collage.paste(img, offset)
    filename = args.save_dir + '/ImageNet_base_test_' + str(iter) + '.jpg'
    collage.save(filename, 'JPEG')
    
if __name__ == '__main__':
    main()