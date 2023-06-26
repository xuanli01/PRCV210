#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import weight_init, conv_bn_relu, upsample_like, ResNet


# Serial Atrous Fusion Module
class SAFM(nn.Module):
    def __init__(self):
        super(SAFM, self).__init__()
        self.cbr1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=2, dilation=2)
        self.cbr2 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=2, dilation=2)
        self.cbr3 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=2, dilation=2)
        self.conv = nn.Conv2d(64 * 4, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, in_f):
        out_f1 = self.cbr1(in_f)
        out_f2 = self.cbr2(out_f1)
        out_f3 = self.cbr3(out_f2)
        out = F.relu(self.bn(self.conv(torch.cat((in_f, out_f1, out_f2, out_f3), dim=1))) + in_f, inplace=True)
        return out

    def initialize(self):
        weight_init(self)


class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel // 4, 1, 1, 0)
        self.conv2 = nn.Conv2d(channel // 4, channel, 1, 1, 0)

    def forward(self, in_f):
        att = nn.AdaptiveAvgPool2d((1, 1))(in_f)
        att = self.conv2(F.relu(self.conv1(att), inplace=True))
        att = nn.Softmax(dim=1)(att)
        att = att - att.min()
        att = att / att.max()
        return att

    def initialize(self):
        weight_init(self)


# Bottom-up detail refinement module
class BDRM(nn.Module):
    def __init__(self):
        super(BDRM, self).__init__()
        self.cbr1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.cbr2 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.ca   = CAM(64 * 2)
        self.conv = nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(64)

    def forward(self, in_high, in_low, downfactor=2):
        fea_low  = self.cbr1(in_low)
        fea_h2l  = self.cbr2(F.max_pool2d(in_high, kernel_size=downfactor, stride=downfactor))
        fea_ca = torch.cat((fea_low, fea_h2l), dim=1)
        att_ca = self.ca(fea_ca)
        fea_fuse = torch.mul(att_ca, fea_ca)
        fea_out  = F.relu(self.bn(self.conv(fea_fuse)) + in_low, inplace=True)
        return fea_out, att_ca, fea_ca

    def initialize(self):
        weight_init(self)


class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv11 = nn.Conv2d(64, 16, kernel_size=(1, 9), padding=(0, 4))
        self.conv21 = nn.Conv2d(64, 16, kernel_size=(9, 1), padding=(4, 0))
        self.conv12 = nn.Conv2d(16, 1, kernel_size=(9, 1), padding=(4, 0))
        self.conv22 = nn.Conv2d(16, 1, kernel_size=(1, 9), padding=(0, 4))
        self.bn1    = nn.BatchNorm2d(16)
        self.bn2    = nn.BatchNorm2d(16)

    def forward(self, in_f):
        b, c, h, w = in_f.shape
        att1 = self.conv12(F.relu(self.bn1(self.conv11(in_f)), inplace=True))
        att2 = self.conv22(F.relu(self.bn2(self.conv21(in_f)), inplace=True))
        att = att1 + att2
        att = torch.sigmoid(att)
        att = att.view(b, 1, h * w)
        att = nn.Softmax(dim=2)(att)
        att = att - att.min()
        att = att / att.max()
        att = att.view(b, 1, h, w)
        return att

    def initialize(self):
        weight_init(self)

# Top-down location refinement module
class TLRM(nn.Module):
    def __init__(self):
        super(TLRM, self).__init__()
        self.cbr1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.cbr2 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.sa   = SAM()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(64)

    def forward(self, in_high, in_low):
        fea_low  = self.cbr1(in_low)
        fea_high = self.cbr2(in_high)
        att_low  = self.sa(fea_low)
        att_l2h  = upsample_like(att_low, in_high)
        fea_fuse = fea_high * att_l2h
        fea_out  = F.relu(self.bn(self.conv(fea_fuse)) + in_high, inplace=True)
        return fea_out, att_l2h

    def initialize(self):
        weight_init(self)

# Attention-guided Bi-directional Feature Refinement Module
class ABFRM(nn.Module):
    def __init__(self):
        super(ABFRM, self).__init__()
        self.ressam1 = TLRM()
        self.ressam2 = TLRM()
        self.ressam3 = TLRM()
        self.rescam1 = BDRM()
        self.rescam2 = BDRM()
        self.rescam3 = BDRM()

    def forward(self, feature):
        f1, f2, f3, f4 = feature[0], feature[1], feature[2], feature[3]
        f1_sa, sa2 = self.ressam3(f1, f2)
        f2_sa, sa3 = self.ressam2(f2, f3)
        f3_sa, sa4 = self.ressam1(f3, f4)
        f2_ca, ca1, ca1f = self.rescam1(f1_sa, f2_sa, 2)
        f3_ca, ca2, ca2f = self.rescam2(f2_sa, f3_sa, 2)
        f4_ca, ca3, ca3f = self.rescam3(f3_sa, f4, 2)
        return (f1_sa, f2_ca, f3_ca, f4_ca), (sa2, sa3, sa4)

    def initialize(self):
        weight_init(self)

    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.msfe1 = SAFM()
        self.msfe2 = SAFM()
        self.msfe3 = SAFM()
        self.msfe4 = SAFM()

    def forward(self, feature):
        f1, f2, f3, f4 = feature[0], feature[1], feature[2], feature[3]
        fb4 = self.msfe4(f4)
        fb3 = self.msfe3(f3 + upsample_like(fb4, f3))
        fb2 = self.msfe2(f2 + upsample_like(fb3, f2))
        fb1 = self.msfe1(f1 + upsample_like(fb2, f1))
        return (fb1, fb2, fb3, fb4)

    def initialize(self):
        weight_init(self)

# Upsampling Feature Refinement Module
class UFRM(nn.Module):
    def __init__(self):
        super(UFRM, self).__init__()
        self.deConv_1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, groups=64)
        self.fuseCbr1 = conv_bn_relu(64, 64, kernel=3, padding=1, stride=1)
        self.fuseCbr2 = conv_bn_relu(64, 64, kernel=3, padding=1, stride=1)
        self.conv = nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, in_f):
        fea1 = F.interpolate(in_f, scale_factor=2, mode='bilinear', align_corners=True)
        fea2 = self.deConv_1(in_f)
        fuse = self.fuseCbr1(fea2)
        fuse = self.fuseCbr2(fuse)
        out = F.relu(self.bn(self.conv(torch.cat((fea2, fuse), dim=1))) + fea1, inplace=True)
        return out

    def initialize(self):
        weight_init(self)


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.bkbone = ResNet()
        self.tran1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.tran2 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.tran3 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.tran4 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.middleLayer = ABFRM()
        self.decoder = Decoder()
        self.p = UFRM()
        self.spv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # ---------- supervision ----------
        self.spv1 = nn.Conv2d(64, 1, 3, 1, 1)
        self.spv2 = nn.Conv2d(64, 1, 3, 1, 1)
        self.spv3 = nn.Conv2d(64, 1, 3, 1, 1)
        self.spv4 = nn.Conv2d(64, 1, 3, 1, 1)

        self.initialize()

    def forward(self, image, shape=None):
        if shape is None:
            shape = image.size()[2:]

        bkbone_f0, bkbone_f1, bkbone_f2, bkbone_f3, bkbone_f4 = self.bkbone(image)
        tran_f1 = self.tran1(bkbone_f1)
        tran_f2 = self.tran2(bkbone_f2)
        tran_f3 = self.tran3(bkbone_f3)
        tran_f4 = self.tran4(bkbone_f4)

        fmid, sa = self.middleLayer((tran_f1, tran_f2, tran_f3, tran_f4))
        fout = self.decoder(fmid)
        final = self.p(fout[0])
        spv_fuse = upsample_like(self.spv(final), shape=shape)
        spv_1 = upsample_like(self.spv1(fout[0]), shape=shape)
        spv_2 = upsample_like(self.spv2(fout[1]), shape=shape)
        spv_3 = upsample_like(self.spv3(fout[2]), shape=shape)
        spv_4 = upsample_like(self.spv4(fout[3]), shape=shape)
        sa1 = upsample_like(sa[0], shape=shape)
        sa2 = upsample_like(sa[1], shape=shape)
        sa3 = upsample_like(sa[2], shape=shape)

        return (spv_fuse, spv_1, spv_2, spv_3, spv_4), (sa1, sa2, sa3)

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot, map_location='cuda:0'))
        else:
            weight_init(self)

