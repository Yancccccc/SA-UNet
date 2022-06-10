import torch
from torch import nn
from dataset import *
import torch.nn.functional as F
#=================================

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=12,dilation=12)
        self.conv3 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv4 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=18, dilation=18)
        self.conv5 = nn.Conv2d(in_channels=out_ch*5,out_channels=out_ch,kernel_size=1,stride=1,padding=0)
        self.asppool = ASPPPooling(in_ch,out_ch)
        self.batchnorm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x1 = self.conv1(x)

        x1 = self.relu(x1)
        x2 = self.conv2(x)

        x2 = self.relu(x2)
        x3 = self.conv3(x)

        x3 = self.relu(x3)
        x4 = self.conv4(x)

        x4 = self.relu(x4)
        x5 =self.asppool(x)
        x5 = self.relu(x5)
        x_cat = torch.cat([x1,x2,x3,x4,x5],dim=1)
        aspp_out = self.conv5(x_cat)
        return aspp_out

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1,padding=1),
            #nn.BatchNorm2d(out_ch),
            #nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class basicconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(basicconv, self).__init__()
        self.conv1 = DoubleConv2(in_ch, out_ch)
        self.aspp1 = ASPP(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1,padding=0)
        self.batchnorm = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        c1 = self.conv1(x)
        c1_aspp = self.aspp1(x)
        c1_add = c1+c1_aspp
        c1 = self.batchnorm(c1_add)
        c_out = self.relu(c1)

        return c_out

class TransConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(TransConv, self).__init__()
        self.tranconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,input):
        return self.tranconv(input)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.sa = SpatialAttention()

    def forward(self,sp,se):
        sp_att = self.sa(sp)
        out = se*sp_att+se
        return out


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.sam = SAM()
        self.conv1 = basicconv(in_ch,64)
        self.conv2 = basicconv(64,128)
        self.conv3 = basicconv(128,256)
        self.conv4 = basicconv(256,512)
        self.conv5 = basicconv(512,1024)
        #self.conv2 = DoubleConv(64, 128)

        # self.conv3 = DoubleConv(128, 256)
        #
        # self.conv4 = DoubleConv(256, 512)
        #
        # self.conv5 = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2,stride=2)
        #self.drop = nn.Dropout2d(p=0.5)
        self.drop = nn.Dropout2d(p=0.5)


        self.up6 = TransConv(1024, 512)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = TransConv(512, 256)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = TransConv(256, 128)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = TransConv(128, 64)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
      a = self.conv1(x)
      a_out = self.pool(a)

      b = self.conv2(a_out)
      b_out = self.pool(b)


      c = self.conv3(b_out)
      c_out = self.pool(c)

      d = self.conv4(c_out)
      d_out = self.pool(d)

      d_out = self.drop(d_out)

      e = self.conv5(d_out)

      e_out = self.drop(e)

      up_6 = self.up6(e_out)# 32 512

      up_6 = self.sam(d, up_6)
      merge6 = torch.cat([up_6, d], dim=1)  #32 1024
      c6 = self.conv6(merge6)
      up_7 = self.up7(c6)

      up_7 = self.sam(c,up_7)
      merge7 = torch.cat([up_7,c], dim=1)
      c7 = self.conv7(merge7)
      up_8 = self.up8(c7)

      up_8 = self.sam(b,up_8)
      merge8 = torch.cat([up_8, b], dim=1)
      c8 = self.conv8(merge8)
      up_9 = self.up9(c8)

      up_9 = self.sam(a,up_9)
      merge9 = torch.cat([up_9, a], dim=1)
      c9 = self.conv9(merge9)
      c10 = self.conv10(c9)

      return c10

#==================================================================================
if __name__ == '__main__':
    from torchsummary import summary
    unet = Unet(11, 8).cuda()
    summary(unet, (11, 256, 256))  # Total params:  5,267,810



















