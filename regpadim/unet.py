import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv as conv


class Gradient_Loss(nn.Module):
    def __init__(self, channels=1, alpha=1, gsize=5):
        super(Gradient_Loss, self).__init__()
        self.alpha = alpha
        filter = torch.FloatTensor([[-1., 1.]])
        self.filter_x = filter.view(1, 1, 1, 2).repeat(1, channels, 1, 1)
        self.filter_y = filter.view(1, 1, 2, 1).repeat(1, channels, 1, 1)

        def gkern(size=5):
            std = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
            n = torch.arange(0, size) - (size - 1.0) / 2.0
            gker1d = torch.exp(-n ** 2 / 2 * std ** 2)
            gker2d = torch.outer(gker1d, gker1d)
            return (gker2d / gker2d.sum())[None, None]

        self.filter_g = gkern(gsize).repeat(channels, 1, 1, 1)
        self.gpad=gsize//2
        self.c = channels

    def forward(self, gen_frames, gt_frames):
        gen_frames = nn.functional.pad(gen_frames, (self.gpad, self.gpad, self.gpad, self.gpad))
        gen_frames = nn.functional.conv2d(gen_frames, self.filter_g, groups=self.c)
        gt_frames = nn.functional.pad(gt_frames, (self.gpad, self.gpad, self.gpad, self.gpad))
        gt_frames = nn.functional.conv2d(gt_frames, self.filter_g, groups=self.c)

        gen_frames_x = nn.functional.pad(gen_frames, (1, 0, 0, 0))
        gen_frames_y = nn.functional.pad(gen_frames, (0, 0, 1, 0))
        gt_frames_x = nn.functional.pad(gt_frames, (1, 0, 0, 0))
        gt_frames_y = nn.functional.pad(gt_frames, (0, 0, 1, 0))
        gen_dx = nn.functional.conv2d(gen_frames_x, self.filter_x)
        gen_dy = nn.functional.conv2d(gen_frames_y, self.filter_y)
        gt_dx = nn.functional.conv2d(gt_frames_x, self.filter_x)
        gt_dy = nn.functional.conv2d(gt_frames_y, self.filter_y)

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return grad_diff_x ** self.alpha + grad_diff_y ** self.alpha


class Smooth_Loss(nn.Module):
    def __init__(self, channels=2, ks=3, alpha=1):
        super(Smooth_Loss, self).__init__()
        self.alpha = alpha
        self.ks = ks
        self.c = channels
        filterx = torch.FloatTensor([[-1 / (ks - 1)] * ks]*1)
        filterx[:, ks//2] = 1
        filtery = torch.FloatTensor([[-1 / (ks - 1)] * 1] * ks)
        filtery[ks // 2, :] = 1

        self.filter_x = filterx.view(1, 1, 1, ks).repeat(channels, 1, 1, 1)
        self.filter_y = filtery.view(1, 1, ks, 1).repeat(channels, 1, 1, 1)
        self.padding = ks//2

    def forward(self, gen_frames):
        gen_frames_x = nn.functional.pad(gen_frames, (self.padding, self.padding, 0, 0))
        gen_frames_y = nn.functional.pad(gen_frames, (0, 0, self.padding, self.padding))
        gen_dx = nn.functional.conv2d(gen_frames_x, self.filter_x, groups=self.c)
        gen_dy = nn.functional.conv2d(gen_frames_y, self.filter_y, groups=self.c)
        return (gen_dx.abs() ** self.alpha + gen_dy.abs() ** self.alpha).mean()



class Encoder(torch.nn.Module):
    def __init__(self, c):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True)
            )
        def CoordBasic(intInput, intOutput):
            return torch.nn.Sequential(
                CoordConv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True),
                CoordConv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True)
            )
        
        self.moduleConv1 = CoordBasic(2*c, 16)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(16, 32)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(32, 32)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(32, 32)
        self.modulePool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(32, 32)
        
    def forward(self, x):

        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        
        return tensorConv5, tensorConv1, tensorConv2, tensorConv3, tensorConv4

    
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True)
            )
                
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias=False),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True)
            )

        self.moduleUpsample5 = Upsample(32, 32)

        self.moduleDeconv4 = Basic(64, 32)
        self.moduleUpsample4 = Upsample(32, 32)

        self.moduleDeconv3 = Basic(64, 32)
        self.moduleUpsample3 = Upsample(32, 32)

        self.moduleDeconv2 = Basic(64, 32)
        self.moduleUpsample2 = Upsample(32, 32)

        self.moduleGenfea = Gen(32,2,32)
        self.moduleGen4 = Gen(64, 2, 32)

    def forward(self, x, skip1, skip2, skip3, skip4):
        tensorUpsample5 = self.moduleUpsample5(x)
        cat5 = torch.cat((skip4, tensorUpsample5), dim=1)

        tensorDeconv4 = self.moduleDeconv4(cat5)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)

        outputfea = self.moduleGenfea(x)
        output4 = self.moduleGen4(cat4)

        return outputfea, output4


class AddCoords2d(nn.Module):
    def __init__(self):
        super(AddCoords2d, self).__init__()
        self.xx_channel = None
        self.yy_channel = None
        self.xx_channel_ = None
        self.yy_channel_ = None

    def forward(self, input_tensor, r=16):
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        if self.xx_channel is None:
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.float32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.float32)
            xx_range = (torch.arange(dim_y, dtype=torch.float32) / (dim_y - 1))
            yy_range = (torch.arange(dim_x, dtype=torch.float32) / (dim_x - 1))
            xx_range_ = (torch.arange(dim_y // r, dtype=torch.float32) / (dim_y // r - 1)).view(dim_y // r, 1).repeat(1, r).flatten()
            yy_range_ = (torch.arange(dim_y // r, dtype=torch.float32) / (dim_y // r - 1)).view(dim_y // r, 1).repeat(1, r).flatten()
            xx_range = xx_range - xx_range_
            yy_range = yy_range - yy_range_

            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]
            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)
            yy_channel = yy_channel.permute(0, 1, 3, 2)
            self.xx_channel = (xx_channel * 2 - 1).to(input_tensor.device)
            self.yy_channel = (yy_channel * 2 - 1).to(input_tensor.device)

            xx_range_ = xx_range_[None, None, :, None]
            yy_range_ = yy_range_[None, None, :, None]
            xx_channel_ = torch.matmul(xx_range_, xx_ones)
            yy_channel_ = torch.matmul(yy_range_, yy_ones)
            yy_channel_ = yy_channel_.permute(0, 1, 3, 2)
            self.xx_channel_ = (xx_channel_ * 2 - 1).to(input_tensor.device)
            self.yy_channel_ = (yy_channel_ * 2 - 1).to(input_tensor.device)
        out = torch.cat([input_tensor, self.xx_channel.repeat(batch_size_shape, 1, 1, 1), self.xx_channel_.repeat(batch_size_shape, 1, 1, 1), self.yy_channel.repeat(batch_size_shape, 1, 1, 1), self.yy_channel_.repeat(batch_size_shape, 1, 1, 1)], dim=1)
        return out


class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.addcoords2d = AddCoords2d()
        self.conv = nn.Conv2d(in_channels + 4, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, input_tensor):
        out = self.addcoords2d(input_tensor)
        out = self.conv(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, inc, outc):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.downsample = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=1, bias=False), nn.BatchNorm2d(outc))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(identity)
        out = self.relu(out)

        return out
    


class unet(torch.nn.Module):
    def __init__(self, size, c=1):
        super(unet, self).__init__()
        self.encoder = Encoder(c)
        self.decoder = Decoder()
        self.resblock5 = nn.Sequential(ResBlock(32, 32), ResBlock(32, 32), ResBlock(32, 32))

        gridY = torch.linspace(-1, 1, steps=size[3]).view(1, 1, -1, 1).expand(1, size[2], size[3], 1)
        gridX = torch.linspace(-1, 1, steps=size[2]).view(1, -1, 1, 1).expand(1, size[2], size[3], 1)
        self.register_buffer('grid', torch.cat((gridX, gridY), dim=3).type(torch.float32))

        self.loss_grad = Gradient_Loss(c)
        self.loss_smooth = Smooth_Loss()
        self.crop_size = [size[0], size[1]]

    def forward(self, x, ref, max_offset1=16, max_offset2=4, test=False):
        ref = ref.repeat((x.size(0), 1, 1, 1))
        bs, c, w, h = x.size()
        fea, skip1, skip2, skip3, skip4 = self.encoder(torch.cat([x, ref], dim=1))
        fea = self.resblock5(fea)
        offset1, offset2 = self.decoder(fea, skip1, skip2, skip3, skip4)

        if not test:
            offset_loss = (((offset1 ** 2).sum(dim=-1) ** 0.5).mean() + ((offset2 ** 2).sum(dim=-1) ** 0.5).mean())/2
            smooth_loss = self.loss_smooth(offset1) + self.loss_smooth(offset2)

        offset1 = offset1 * 2 * max_offset1 / w
        offset2 = offset2 * 2 * max_offset2 / w
        offset1 = F.interpolate(offset1, (w, h), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        offset2 = F.interpolate(offset2, (w, h), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        grid1 = torch.clamp(offset1 + self.grid, min=-1, max=1)
        grid2 = torch.clamp(offset2 + self.grid, min=-1, max=1)
        x_1 = F.grid_sample(x, grid1, align_corners=True)
        x_2 = F.grid_sample(x_1, grid2, align_corners=True)

        if test: return x_2[:, :, self.crop_size[0]:-self.crop_size[0], self.crop_size[1]:-self.crop_size[1]]

        weight = 0.1 + 5 * self.loss_grad(x, ref)[:, :, self.crop_size[0]:-self.crop_size[0], self.crop_size[1]:-self.crop_size[1]].mean((1,2,3),keepdim=True)
        grad_loss = (self.loss_grad(x_2, ref)[:, :, self.crop_size[0]:-self.crop_size[0], self.crop_size[1]:-self.crop_size[1]]*weight).mean() + 0.25 * (self.loss_grad(x_1, ref)[:, :, self.crop_size[0]:-self.crop_size[0], self.crop_size[1]:-self.crop_size[1]]*weight).mean()
        pix_loss = (F.mse_loss(x_2[:, :, self.crop_size[0]:-self.crop_size[0], self.crop_size[1]:-self.crop_size[1]], ref[:, :, self.crop_size[0]:-self.crop_size[0], self.crop_size[1]:-self.crop_size[1]], reduction='none')*weight).mean()

        return x_2[:, :, self.crop_size[0]:-self.crop_size[0], self.crop_size[1]:-self.crop_size[1]], pix_loss, grad_loss, smooth_loss, offset_loss, weight