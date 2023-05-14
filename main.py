from tqdm import tqdm
import math
import torch
import torch.nn as nn
import argparse
import os
import torchvision
import time
import random
import warnings
import numpy as np
import shutil
import wandb
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_AE import *
from dataset import Transform,Transform_noaug, zcaTransformation,addZeroPadding,addPiexlZeroPadding
from utils import AverageMeter

from PIL import Image


parser = argparse.ArgumentParser(description='Autoencoder Training')
# General variables
parser.add_argument('--data', metavar='DIR', default='/scratch/jchen175/projects/nips/imagenette2')
parser.add_argument('--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--save_dir', type=str, default='./experiments/testing/')
parser.add_argument('--workers', type=int, metavar='N', default=8)
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--checkpoint_freq', type=int, default=19)

# Training variables
parser.add_argument('--epochs', type=int, metavar='N', default=400)#100
parser.add_argument('--batch_size', type=int, metavar='N', default=512) # 1024
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, metavar='W', default=0.001)
parser.add_argument('--stop_epoch', type=int, default=None)
parser.add_argument('--no_aug', action='store_true')

parser.add_argument('--noise', action='store_true', help='add noise to input images')
parser.add_argument('--noise_rgb', action='store_true', help='zero padding at R/G/B instaed of pixel (R+G+B)')
parser.add_argument('--noise_prob', type=float, default=0.4, help='prob of pixel been zero padding')
parser.add_argument('--ZCA', action='store_true', help='apply ZCA whitening')
parser.add_argument('--batch_level_zca', action='store_true', help='apply ZCA at batch level')

# SEED
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')

# GPU
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')




def main():
    args = parser.parse_args()
    run_name = f"denoise@{args.noise}_ZCA@{args.ZCA}_batch@{args.batch_size}_lr@{args.lr}_" \
               f"epoch@{args.epochs}_noiseProb@{args.noise_prob}_scheduler@Ture" \
               f"_noise@{'rgb' if args.noise_rgb else 'pixel'}_batch_level_zca@{args.batch_level_zca}"
    experiment = wandb.init(project='resUnetReconstruct', resume='allow', anonymous='must',
                            name=run_name)
    experiment.config.update(
        dict(
            arch = args.arch,
            epochs = args.epochs,
            batch_size = args.batch_size,
            lr = args.lr,
            weight_decay = args.weight_decay,
            noise = args.noise,
            noise_type = 'rgb' if args.noise_rgb else 'pixel',
            noise_prob = args.noise_prob,
            ZCA = args.ZCA,
            seed = args.seed,
            )
    )
    # print(vars(args))

    args.save_dir_models = os.path.join(args.save_dir, run_name, 'ckpt')
    os.makedirs(args.save_dir_models, exist_ok=True)

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
    global_step = 0

    print('\nUsing Single GPU training')
    print('Use GPU: {} for training'.format(args.gpu))

    print('\nLoading dataloader ...')

    transform_funct = Transform_noaug() if args.no_aug else Transform()
    train_dataset = datasets.ImageFolder(os.path.join(args.data,'train'), transform_funct)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True)
    # val_transform = transforms.Compose([
    #                     transforms.Resize(256),
    #                     transforms.CenterCrop(224),
    #                     transforms.ToTensor(),])
    val_transform = Transform_noaug()
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    print('\nStart training...')
    start_epoch=0
    all_losses = []
    for epoch in range(start_epoch, args.epochs):
        start_time_epoch = time.time()

        # train the network
        epoch_loss, global_step = train(
            train_loader,
            model,
            optimizer,
            epoch,
            args,
            global_step,
        )
        all_losses.append(epoch_loss)

        # validation and save reconstruccions
        epoch_val_loss, val_visualization = validate(val_loader, model, args)
        print("Epoch: [{}] | Train loss: {:.4f} | Val loss: {:.4f}".format(epoch, epoch_loss, epoch_val_loss))
        # log loss/ visualizations
        # val image reconstruct:
        val_visualization = torchvision.utils.make_grid(val_visualization, nrow = 20, pad_value=1)
        kernels = model.encoder.conv1.weight.detach().cpu().clone()
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        kernel_visualization = torchvision.utils.make_grid(kernels, nrow=16, pad_value=1)
        scheduler.step(epoch_val_loss)
        wandb.log(
            {
                'train loss': epoch_loss,
                'val loss': epoch_val_loss,
                'epoch': epoch,
                'lr': optimizer.param_groups[0]["lr"],
                'val visualization': wandb.Image(val_visualization),
                'conv1': wandb.Image(kernel_visualization),
            }
        )
        # TODO: resume this after finding the best hyper-par;
        #  I don't have enough quota
        # # save checkpoints and encoder
        # try:
        #     save_checkpoints_encoder(model, optimizer, epoch, args)
        # except:
        #     pass

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


def train(loader, model, optimizer, epoch, args,global_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    criterion = nn.MSELoss().cuda(args.gpu)
    make_noise = addZeroPadding if args.noise_rgb else addPiexlZeroPadding


    end = time.time()
    for it, (inputs, labels) in enumerate(loader):

        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda(args.gpu, non_blocking=True) # used for cal loss
        if args.ZCA:
            if args.noise:
                inputs_ = make_noise(inputs,args)
                inputs_ = zcaTransformation(inputs_,args)
                output = model(inputs_)
            else:
                inputs_ = zcaTransformation(inputs, args)
                output = model(inputs_)
        elif args.noise:
            inputs_ = make_noise(inputs,args)
            output = model(inputs_)
        else:
            output = model(inputs)
        loss = criterion(output, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        try:
            save_checkpoints_encoder(model, optimizer, epoch, args, global_step)
        except:
            pass

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
    return losses.avg, global_step


def validate(val_loader, model, args):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    criterion = nn.MSELoss()

    make_noise = addZeroPadding if args.noise_rgb else addPiexlZeroPadding
    val_visualization = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            # log zca whitened input
            if args.ZCA:
                if args.noise:
                    inputs_ = zcaTransformation(inputs, args)
                    inputs_ = make_noise(inputs_, args)
                    output = model(inputs_)
                else:
                    inputs_ = zcaTransformation(inputs, args)
                    output = model(inputs_)
            elif args.noise:
                inputs_ = make_noise(inputs, args)
                output = model(inputs_)
            else:
                output = model(inputs)
            loss = criterion(output, inputs)

            # misc
            losses.update(loss.item(), inputs.size(0))

            # save reconstructed images
            if (not args.ZCA) and (not args.noise):
                disp = torch.cat((inputs[:20], output[:20]), dim=0)
            else:
                disp = torch.cat((inputs_[:20], output[:20]), dim=0)
            val_visualization.append(disp.cpu())
            # imgdisp(disp.cpu(), i, args)
    val_visualization = val_visualization[:5]
    val_visualization = torch.cat(val_visualization, dim=0)
    return losses.avg, val_visualization



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
    
def save_checkpoints_encoder(model, optimizer, epoch, args,global_step):
    # save autoencoder
    if (global_step+1) % args.checkpoint_freq  == 0:
        save_dict = {"step": global_step + 1,
                    "state_dict": model.state_dict()}
        torch.save(save_dict, os.path.join(args.save_dir_models, "ckp_step_" + str(global_step+1) + ".pth"))

        # save encoder
        save_dict = {"step": global_step + 1,
                     "state_dict": model.encoder.state_dict()}
        torch.save(save_dict,  os.path.join(args.save_dir_models, str(args.arch) + "_step_" + str(global_step+1) + ".pth"))
    else:
        return



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