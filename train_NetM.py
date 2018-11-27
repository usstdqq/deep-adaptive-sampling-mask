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
from model import NetME, NetE
from tensorboard_logger import configure, log_value, log_images

# Training settings
parser = argparse.ArgumentParser(description='PyTorch NetM')
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
parser.add_argument('--modelE_name', default='model_best.pth', type=str, help='model name of the trained NetE')
opt = parser.parse_args()

print(opt)

print('===> Select GPU to train...') 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ]))
    
val_set = DatasetFromH5('data_val_100.h5', 
                        input_transform=transforms.ToTensor(), target_transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)


print('===> Building ME model...')
modelME = NetME(nef = opt.nef, NetE_name = 'epochs_NetE/' + opt.modelE_name, sample_rate = opt.sample_rate)

if torch.cuda.is_available():
    modelME = modelME.cuda()
modelME.netM.train()
modelME.netE.eval()
print(modelME)

criterion = nn.MSELoss()

if torch.cuda.is_available():
    criterion = criterion.cuda()

print('===> Total Model NetME Parameters:', sum(param.numel() for param in modelME.parameters()))

print('===> Initialize Optimizer...')      
optimizer = optim.Adam([{'params': modelME.netM.parameters(), 'lr': opt.lr},
                        {'params': modelME.netE.parameters(), 'lr': 0.0}
                        ], lr=opt.lr)

if not os.path.exists("epochs_NetME"):
        os.makedirs("epochs_NetME")

if not os.path.exists("tensorBoardRuns"):
        os.makedirs("tensorBoardRuns")

print('===> Initialize Logger...')     
configure("tensorBoardRuns/mask-train-conti-on-demand-learn-p-02-zero-corrupt-zero-conv-bias-conti-ber-train-v4-cwfc-one-net-eval-h5-val-sig-M-res_net-clip-mean-iter-switch-epoch-800")


def train(epoch):
    epoch_loss = 0
    epoch_psnr = 0
    epoch_sparsity = 0

    #	train/eval modes make difference on batch normalization layer
    modelME.netM.train()
    modelME.netE.eval()
    
    #   Step up learning rate decay
    #   No learning rate decay here
    #   Learning rate of NetE is fixed to be 0
    lr = opt.lr
    optimizer = optim.Adam([{'params': modelME.netM.parameters(), 'lr': opt.lr},
                            {'params': modelME.netE.parameters(), 'lr': 0.0}
                            ], lr=opt.lr)
    
    for iteration, batch in enumerate(train_loader, 1):
        target, _ = batch
        image = target.clone()       
 
        #	mean_image and std_image are used to compute loss
        mean_image = torch.zeros(image.shape[0], image.shape[1], image.shape[2], image.shape[3])
        mean_image[:,0,:,:] = 0.5
        mean_image[:,1,:,:] = 0.5
        mean_image[:,2,:,:] = 0.5
        
        std_image = torch.zeros(image.shape[0], image.shape[1], image.shape[2], image.shape[3])
        std_image[:,0,:,:] = 0.5
        std_image[:,1,:,:] = 0.5
        std_image[:,2,:,:] = 0.5
        
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()
            mean_image = mean_image.cuda()
            std_image = std_image.cuda()
            
        optimizer.zero_grad()
         
        #   Generate the corruption mask and reconstructed image
        corrupt_mask_conti, image_recon = modelME(image)

        mask_sparsity = corrupt_mask_conti.sum() / (corrupt_mask_conti.shape[0] * corrupt_mask_conti.shape[1] * corrupt_mask_conti.shape[2] * corrupt_mask_conti.shape[3])
        
        loss = criterion((image_recon*std_image)+mean_image, (target*std_image)+mean_image)
        psnr = 10 * log10(1 / loss.data[0])
        epoch_loss += loss.data[0]
        epoch_psnr += psnr
        epoch_sparsity += mask_sparsity
        loss.backward()
        optimizer.step()

    print("===> Epoch {} Complete: lr: {}, Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB, Mask Sparsity: {:.4f}".format(epoch, lr, epoch_loss / len(train_loader), epoch_psnr / len(train_loader), epoch_sparsity / len(train_loader)))
    
    log_value('train_loss', epoch_loss / len(train_loader), epoch)
    log_value('train_psnr', epoch_psnr / len(train_loader), epoch)
    log_value('train_sparsity', epoch_sparsity / len(train_loader), epoch) 

PSNR_best = 0

def reshape_4D_array(array_4D, width_num):
    num, cha, height, width = array_4D.shape
    height_num = num // width_num
    total_width = width * width_num
    total_height = height * height_num
    target_array_4D = np.zeros((1, cha, total_width, total_height))
    for index in range(0, num):
        height_start = index//width_num
        width_start = index%width_num
        target_array_4D[:,:,height_start*height:height_start*height+height,width_start*width:width_start*width+width] = array_4D[index,:,:,:]
    return target_array_4D

def val(epoch):
    avg_psnr = 0
    avg_mse = 0
    avg_sparsity = 0
    
    modelME.eval()
    modelME.netM.eval()
    modelME.netE.eval()

    for batch in val_loader:
        target = batch
        image = target.clone()        
        image_clone = image.clone()
        
        mean_image = torch.zeros(image.shape[0], image.shape[1], image.shape[2], image.shape[3])
        mean_image[:,0,:,:] = 0.5
        mean_image[:,1,:,:] = 0.5
        mean_image[:,2,:,:] = 0.5
        
        std_image = torch.zeros(image.shape[0], image.shape[1], image.shape[2], image.shape[3])
        std_image[:,0,:,:] = 0.5
        std_image[:,1,:,:] = 0.5
        std_image[:,2,:,:] = 0.5
        
        if torch.cuda.is_available():
            image = image.cuda()
            image_clone = image_clone.cuda()
            target = target.cuda()
            mean_image = mean_image.cuda()
            std_image = std_image.cuda()
         
        #   Generate the corruption mask and reconstructed image
        corrupt_mask_conti, image_recon = modelME(image)
        
        corrupt_mask = corrupt_mask_conti.bernoulli()   # Binarize the corruption mask using Bernoulli distribution, then feed into modelE
        mask_sparsity = corrupt_mask.sum() / (corrupt_mask.shape[0] * corrupt_mask.shape[1] * corrupt_mask.shape[2] * corrupt_mask.shape[3])
        corrupt_mask = corrupt_mask.expand(corrupt_mask.shape[0], 3, corrupt_mask.shape[2], corrupt_mask.shape[3])
        
        #   Generate the corrupted image
        mask_image = corrupt_mask * image_clone
        
        restored_image = modelME.netE(mask_image)
        
        mse = criterion((restored_image*std_image)+mean_image, (target*std_image)+mean_image)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        avg_mse  += mse.data[0]
        avg_sparsity += mask_sparsity
        
    print("===> Epoch {} Validation: Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB, Mask Sparsity: {:.4f}".format(epoch, avg_mse / len(val_loader), avg_psnr / len(val_loader), avg_sparsity / len(val_loader)))

    log_value('val_loss', avg_mse / len(val_loader), epoch)
    log_value('val_psnr', avg_psnr / len(val_loader), epoch)
    log_value('val_sparsity', avg_sparsity / len(val_loader), epoch)
    
    corrupt_mask_conti = corrupt_mask_conti.expand(corrupt_mask_conti.shape[0], 3, corrupt_mask_conti.shape[2], corrupt_mask_conti.shape[3])

    log_images('original_image', reshape_4D_array((image*std_image+mean_image).cpu().numpy(), 10), step=1)
    log_images('conti_mask', reshape_4D_array(corrupt_mask_conti.data.cpu().numpy(), 10), step=1)
    log_images('binar_mask', reshape_4D_array(corrupt_mask.data.cpu().numpy(), 10), step=1)
    log_images('restored_image', reshape_4D_array((restored_image*std_image+mean_image).data.cpu().numpy(), 10), step=1)
    
    global PSNR_best
    if avg_psnr > PSNR_best:
        PSNR_best = avg_psnr
        model_out_path = "epochs_NetME/" + "model_best.pth"
        torch.save(modelME.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

def val_rand(epoch):
    avg_psnr = 0
    avg_mse = 0
    avg_sparsity = 0
    
    modelME.eval()
    modelME.netM.eval()
    modelME.netE.eval()
    
    for batch in val_loader:
        target = batch
        image = target.clone()
        
        mean_image = torch.zeros(image.shape[0], image.shape[1], image.shape[2], image.shape[3])
        mean_image[:,0,:,:] = 0.5
        mean_image[:,1,:,:] = 0.5
        mean_image[:,2,:,:] = 0.5
        
        std_image = torch.zeros(image.shape[0], image.shape[1], image.shape[2], image.shape[3])
        std_image[:,0,:,:] = 0.5
        std_image[:,1,:,:] = 0.5
        std_image[:,2,:,:] = 0.5

        #   Generate the random corruption mask
        corrupt_mask = torch.ones(image.shape[0], 1, image.shape[2], image.shape[3])
        corrupt_mask = corrupt_mask * opt.sample_rate
        mask_sparsity = corrupt_mask.sum() / (corrupt_mask.shape[0] * corrupt_mask.shape[1] * corrupt_mask.shape[2] * corrupt_mask.shape[3])
        corrupt_mask = corrupt_mask.bernoulli()
        corrupt_mask = corrupt_mask.expand(corrupt_mask.shape[0], 3, corrupt_mask.shape[2], corrupt_mask.shape[3])
        
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()
            mean_image = mean_image.cuda()
            std_image = std_image.cuda()
            corrupt_mask = corrupt_mask.cuda()
        
        #   Generate the corrupted image
        mask_image = corrupt_mask * image
        
        mse = criterion(( modelME.netE(mask_image)*std_image)+mean_image, (target*std_image)+mean_image)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        avg_mse  += mse.data[0]

    print("===> Epoch {} Random Validation: Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB, Mask Sparsity: {:.4f}".format(epoch, avg_mse / len(val_loader), avg_psnr / len(val_loader), avg_sparsity / len(val_loader)))

    log_value('val_loss_rand', avg_mse / len(val_loader), epoch)
    log_value('val_psnr_rand', avg_psnr / len(val_loader), epoch)
    log_value('val_sparsity_rand', avg_sparsity / len(val_loader), epoch)

def checkpoint(epoch):
    if epoch%100 == 0:
        model_out_path = "epochs_NetME/" + "model_epoch_{}.pth".format(epoch)
        torch.save(modelME.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

val(0)
val_rand(0)
checkpoint(0)
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    val(epoch)
    val_rand(epoch)
    checkpoint(epoch)
