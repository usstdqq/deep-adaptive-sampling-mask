from __future__ import print_function
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import numpy as np
from math import log10
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_utils import DatasetFromH5
from model import NetE
from tensorboard_logger import configure, log_value


# Training settings
parser = argparse.ArgumentParser(description='PyTorch NetE')
parser.add_argument('--sample_rate', type=int, default=0.2, help="sample rate for pixel interpolation")
parser.add_argument('--batchSize', type=int, default=128, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=800, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum term. Default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--nef', type=int, default=64, help='number of encoder filters in first conv layer')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
opt = parser.parse_args()

print(opt)

print('===> Select GPU to train...') 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets...')
#	To achieve fast data IO speed, it is suggested to put the training data on a SSD (solid state drive)
train_set = dset.ImageFolder(root='/home/dqq/AIC_XRF_Inpainting/mydataset/my_train_set/',
                             transform=transforms.Compose([
                                     transforms.RandomCrop(size = opt.imageSize, pad_if_needed = True),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ]))

val_set = DatasetFromH5('data_val_100.h5', 
                        input_transform=transforms.ToTensor(), target_transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)



print('===> Building model...')
model = NetE(nef = opt.nef)
criterion = nn.MSELoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
print(model)


print('===> Total Model NetE Parameters:', sum(param.numel() for param in model.parameters()))


print('===> Initialize Optimizer...')      
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if not os.path.exists("epochs_NetE"):
        os.makedirs("epochs_NetE")

if not os.path.exists("tensorBoardRuns"):
        os.makedirs("tensorBoardRuns")

print('===> Initialize Logger...')     
configure("tensorBoardRuns/on-demand-learn-p-02-zero-corrupt-0-conv-bias-0-cwfc-epoch-800")





def train(epoch):
    epoch_loss = 0
    epoch_psnr = 0
    
    #   Step up learning rate decay
    lr = opt.lr
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    
    for iteration, batch in enumerate(train_loader, 1):
        target, _ = batch
        batch_size = target.size(0)
        image = target.clone()
        #   Corrupt the target image
        for i in range(0, batch_size):
            corrupt_mask = np.random.binomial(1, (1 - opt.sample_rate), (opt.imageSize, opt.imageSize))
            corrupt_mask.astype(np.uint8)
            corrupt_mask = torch.ByteTensor(corrupt_mask)
            
            image[i,0,:,:].masked_fill_(corrupt_mask, (0.0))
            image[i,1,:,:].masked_fill_(corrupt_mask, (0.0))
            image[i,2,:,:].masked_fill_(corrupt_mask, (0.0))
            
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        loss = criterion((model(image)+1.0)/2.0, (target+1.0)/2.0)
        psnr = 10 * log10(1 / loss.data[0])
        epoch_loss += loss.data[0]
        epoch_psnr += psnr
        loss.backward()
        optimizer.step()

    print("===> Epoch {} Complete: lr: {}, Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB".format(epoch, lr, epoch_loss / len(train_loader), epoch_psnr / len(train_loader)))
    
    log_value('train_loss', epoch_loss / len(train_loader), epoch)
    log_value('train_psnr', epoch_psnr / len(train_loader), epoch)
    

PSNR_best = 0

def val(epoch):
    avg_psnr = 0
    avg_mse = 0
    for batch in val_loader:
        target = batch
        batch_size = target.size(0)
        image = target.clone()
        #   Corrupt the target image
        for i in range(0, batch_size):
            corrupt_mask = np.random.binomial(1, (1 - opt.sample_rate), (opt.imageSize, opt.imageSize))
            corrupt_mask.astype(np.uint8)
            corrupt_mask = torch.ByteTensor(corrupt_mask)
            
            image[i,0,:,:].masked_fill_(corrupt_mask, (0.0))
            image[i,1,:,:].masked_fill_(corrupt_mask, (0.0))
            image[i,2,:,:].masked_fill_(corrupt_mask, (0.0))
            
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

        prediction = model(image)
        mse = criterion((prediction+1.0)/2.0, (target+1.0)/2.0)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        avg_mse  += mse.data[0]
    print("===> Epoch {} Validation: Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB".format(epoch, avg_mse / len(val_loader), avg_psnr / len(val_loader)))

    log_value('val_loss', avg_mse / len(val_loader), epoch)
    log_value('val_psnr', avg_psnr / len(val_loader), epoch)
    
    global PSNR_best
    if avg_psnr > PSNR_best:
        PSNR_best = avg_psnr
        model_out_path = "epochs_NetE/" + "model_best.pth".format(epoch)
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

def checkpoint(epoch):
    if epoch%100 == 0:
        model_out_path = "epochs_NetE/" + "model_epoch_{}.pth".format(epoch)
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

val(0)
checkpoint(0)
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    val(epoch)
    checkpoint(epoch)
