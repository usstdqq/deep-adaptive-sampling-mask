import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
import torch
import math

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
    

class NetE(nn.Module):
    def __init__(self, nef):
        super(NetE, self).__init__()
        # state size: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(3, nef, (4, 4), (2, 2), (1, 1), bias=False)
        self.conv1_bn = nn.BatchNorm2d(nef)
        self.conv1_relu = nn.LeakyReLU(0.2, inplace=False)
        # state size: (nef) x 32 x 32
        self.conv2 = nn.Conv2d(nef, nef*2, (4, 4), (2, 2), (1, 1), bias=False)
        self.conv2_bn = nn.BatchNorm2d(nef*2)
        self.conv2_relu = nn.LeakyReLU(0.2, inplace=False)
        # state size: (nef*2) x 16 x 16
        self.conv3 = nn.Conv2d(nef*2, nef*4, (4, 4), (2, 2), (1, 1), bias=False)
        self.conv3_bn = nn.BatchNorm2d(nef*4)
        self.conv3_relu = nn.LeakyReLU(0.2, inplace=False)
        # state size: (nef*4) x 8 x 8
        self.conv4 = nn.Conv2d(nef*4, nef*8, (4, 4), (2, 2), (1, 1), bias=False)
        self.conv4_bn = nn.BatchNorm2d(nef*8)
        self.conv4_relu = nn.LeakyReLU(0.2, inplace=False)
        # state size: (nef*8) x 4 x 4

        # channel-wise fully connected layer
        self.channel_wise_layers = []
        
        for i in range(0, 512):
            self.add_module('channel_wise_layers_' + str(i), nn.Linear(16, 16))
        
        self.channel_wise_layers = AttrProxy(self, 'channel_wise_layers_')
            
        # state size: (nef*8) x 4 x 4
        self.dconv1 = nn.ConvTranspose2d(nef*8, nef*4, (4, 4), (2, 2), (1, 1), bias=False)
        self.dconv1_bn = nn.BatchNorm2d(nef*4)
        self.dconv1_relu = nn.ReLU(inplace=True)
        # state size: (nef*4) x 8 x 8
        self.dconv2 = nn.ConvTranspose2d(nef*4, nef*2, (4, 4), (2, 2), (1, 1), bias=False)
        self.dconv2_bn = nn.BatchNorm2d(nef*2)
        self.dconv2_relu = nn.ReLU(inplace=True)
        # state size: (nef*2) x 16 x 16
        self.dconv3 = nn.ConvTranspose2d(nef*2, nef, (4, 4), (2, 2), (1, 1), bias=False)
        self.dconv3_bn = nn.BatchNorm2d(nef)
        self.dconv3_relu = nn.ReLU(inplace=True)
        # state size: (nef) x 32 x 32
        self.dconv4 = nn.ConvTranspose2d(nef, 3, (4, 4), (2, 2), (1, 1), bias=False)
        self.dconv4_tanh = nn.Tanh()
        # self.dconv1_bn = nn.BatchNorm2d(3)
        # state size: (nc) x 64 x 64
        
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1_relu(self.conv1_bn(self.conv1(x)))
        x = self.conv2_relu(self.conv2_bn(self.conv2(x)))
        x = self.conv3_relu(self.conv3_bn(self.conv3(x)))
        x = self.conv4_relu(self.conv4_bn(self.conv4(x)))

        for i in range(0, 512):
            slice_cur = x[:,[i],:,:]
            slice_cur_size = slice_cur.size()
            slice_cur = slice_cur.view(slice_cur_size[0], slice_cur_size[2]*slice_cur_size[3])
            slice_cur = self.channel_wise_layers[i](slice_cur)
            x[:,[i],:,:] = slice_cur.view(slice_cur_size[0], slice_cur_size[1], slice_cur_size[2], slice_cur_size[3])

        x = self.dconv1_relu(self.dconv1_bn(self.dconv1(x)))
        x = self.dconv2_relu(self.dconv2_bn(self.dconv2(x)))
        x = self.dconv3_relu(self.dconv3_bn(self.dconv3(x)))
        x = self.dconv4_tanh(self.dconv4(x))
        return x
    
    def _initialize_weights(self):
        
        init.normal_(self.conv1_bn.weight,  1.0, 0.02)
        init.normal_(self.conv2_bn.weight,  1.0, 0.02)
        init.normal_(self.conv3_bn.weight,  1.0, 0.02)
        init.normal_(self.conv4_bn.weight,  1.0, 0.02)
        init.normal_(self.dconv1_bn.weight, 1.0, 0.02)
        init.normal_(self.dconv2_bn.weight, 1.0, 0.02)
        init.normal_(self.dconv3_bn.weight, 1.0, 0.02)
        
        init.constant_(self.conv1_bn.bias,    0.0)
        init.constant_(self.conv2_bn.bias,    0.0)
        init.constant_(self.conv3_bn.bias,    0.0)
        init.constant_(self.conv4_bn.bias,    0.0)
        init.constant_(self.dconv1_bn.bias,   0.0)
        init.constant_(self.dconv2_bn.bias,   0.0)
        init.constant_(self.dconv3_bn.bias,   0.0)
        
        init.normal_(self.conv1.weight,  0.0, 0.02)
        init.normal_(self.conv2.weight,  0.0, 0.02)
        init.normal_(self.conv3.weight,  0.0, 0.02)
        init.normal_(self.conv4.weight,  0.0, 0.02)
        init.normal_(self.dconv1.weight, 0.0, 0.02)
        init.normal_(self.dconv2.weight, 0.0, 0.02)
        init.normal_(self.dconv3.weight, 0.0, 0.02)
        init.normal_(self.dconv4.weight, 0.0, 0.02)

# NetE ends here
# NetM starts here     
class Mean_Shift(nn.Module):
    def __init__(self, sample_rate=0.2):
        super(Mean_Shift, self).__init__()
        self.sample_rate = sample_rate
        self.sample_rate = torch.autograd.Variable(torch.tensor(sample_rate), requires_grad=False)
        if torch.cuda.is_available(): self.sample_rate = self.sample_rate.cuda()
	
    def forward(self, x):
        x_size = x.size()
        
        x_mean = torch.mean(x, 2, True)
        x_mean = torch.mean(x_mean, 3, True)
        x_mean = x_mean.expand(x_size[0], x_size[1], x_size[2], x_size[3])
        
        x_out = x / x_mean * self.sample_rate
        
        return x_out


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 


class NetM(nn.Module):
    def __init__(self, nef, sample_rate):
        super(NetM, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual = self.make_layer(_Residual_Block, 8)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=1, padding=4, bias=False)
        self.conv_output_bn = nn.BatchNorm2d(1)
        self.conv_output_sig = nn.Sigmoid()
        
        self.mean_shift = Mean_Shift(sample_rate=sample_rate)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        
        out = self.conv_output_sig(self.conv_output_bn(self.conv_output(out)))
        out = self.mean_shift(out)
        
	# iterative mean-clamp, to keep the sample rate precise
        for i in range(0, 25):
            #   Clip to [0, 1]
            out = torch.clamp(out, min=0.0, max=1.0)
            out = self.mean_shift(out)
        
        return out

class NetME(nn.Module):
    def __init__(self, nef, NetE_name, sample_rate):
        super(NetME, self).__init__()
        self.netM  = NetM(nef = 64, sample_rate = sample_rate)
        self.netE = NetE(nef = 64)
        self.netE = torch.load(NetE_name)

    def forward(self, x):
        x_clone = x.clone()
        mask = self.netM(x)
        mask_4d = mask.expand(mask.shape[0], 3, mask.shape[2], mask.shape[3])
        
        mask_x = mask_4d * x_clone
        x_recon = self.netE(mask_x)
        
        return mask, x_recon
       
if __name__ == "__main__":
    model = NetE(nef=64)
    print(model)
