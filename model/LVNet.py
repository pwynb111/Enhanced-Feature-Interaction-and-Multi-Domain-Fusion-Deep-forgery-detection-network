import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("..")
from components.attention import ChannelAttention, SpatialAttention
from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from components.linear_fusion import HdmProdBilinearFusion
from model.xception import TransferModel
from model.modules import *


class Two_Stream_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = {
            'location': {
                'size': 19,
                'channels': [64, 128, 256, 728, 728, 728],
                'mid_channel': 512
            },
            'cls_size': 10,
            'HBFusion': {
                'hidden_dim': 2048,
                'output_dim': 4096,
            }
        }
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.relu = nn.ReLU(inplace=True)

        self.score0 = BasicConv2d(self.params['location']['channels'][0],
                                  self.params['location']['mid_channel'], kernel_size=1)
        self.score1 = BasicConv2d(self.params['location']['channels'][1],
                                  self.params['location']['mid_channel'], kernel_size=1)
        self.score2 = BasicConv2d(self.params['location']['channels'][2],
                                  self.params['location']['mid_channel'], kernel_size=1)
        self.score3 = BasicConv2d(self.params['location']['channels'][3],
                                  self.params['location']['mid_channel'], kernel_size=1)
        self.score4 = BasicConv2d(self.params['location']['channels'][4],
                                  self.params['location']['mid_channel'], kernel_size=1)
        self.score5 = BasicConv2d(self.params['location']['channels'][5],
                                  self.params['location']['mid_channel'], kernel_size=1)

        self.msff = EMFF(in_channels=self.params['location']['mid_channel'],
                         size=self.params['location']['size'])
        self.HBFusion = HdmProdBilinearFusion(dim1=(64 + 128 + 256 + 728 + 728), dim2=2048,
                                              hidden_dim=self.params['HBFusion']['hidden_dim'],
                                              output_dim=self.params['HBFusion']['output_dim'])

        self.cfie0 = CFIE(in_channel=self.params['location']['channels'][0])
        self.cfie1 = CFIE(in_channel=self.params['location']['channels'][1])
        self.cfie2 = CFIE(in_channel=self.params['location']['channels'][2])

        self.lfe0 = EFG(in_channel=self.params['location']['channels'][3])
        self.lfe1 = EFG(in_channel=self.params['location']['channels'][4])
        self.lfe2 = EFG(in_channel=self.params['location']['channels'][5])

        self.cls_header = nn.Sequential(
            nn.BatchNorm2d(self.params['HBFusion']['output_dim']),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(self.params['HBFusion']['output_dim'], 2),
        )
        seg_in_channels = self.params['location']['mid_channel'] * 6
        self.seg_header = nn.Sequential(
            nn.BatchNorm2d(seg_in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(seg_in_channels, 2, kernel_size=1, bias=False),
        )

    def pad_max_pool(self, x):
        b, c, h, w = x.size()
        padding = abs(h % self.params['cls_size'] - self.params['cls_size']) % self.params['cls_size']
        pad = nn.ReplicationPad2d(padding=(padding // 2, (padding + 1) // 2, padding // 2, (padding + 1) // 2)).to(
            x.device)
        x = pad(x)
        b, c, h, w = x.size()

        max_pool = nn.MaxPool2d(kernel_size=h // self.params['cls_size'], stride=h // self.params['cls_size'],
                                padding=0)
        return max_pool(x)

    def get_mask(self, mask):
        b, c, h, w = mask.size()
        mask = mask.float()
        padding = abs(h % self.params['location']['size'] - self.params['location']['size']) % self.params['location'][
            'size']
        pad = nn.ReplicationPad2d(padding=(padding // 2, (padding + 1) // 2, padding // 2, (padding + 1) // 2)).to(
            mask.device)
        max_pool = nn.MaxPool2d(kernel_size=h // self.params['location']['size'],
                                stride=h // self.params['location']['size'], padding=0)

        return max_pool(mask)

    def features(self, x):
        srm = self.srm_conv0(x)

        # 64 * 150 * 150
        x0 = self.xception_rgb.model.fea_part1_0(x)
        y0 = self.xception_srm.model.fea_part1_0(srm)
        x0, y0 = self.cfie0(x0, y0)

        # 128 * 75 * 75
        x1 = self.xception_rgb.model.fea_part1_1(x0)
        y1 = self.xception_srm.model.fea_part1_1(y0)
        x1, y1 = self.cfie1(x1, y1)

        # 256 * 38 * 38
        x2 = self.xception_rgb.model.fea_part1_2(x1)
        y2 = self.xception_srm.model.fea_part1_2(y1)
        x2, y2 = self.cfie2(x2, y2)

        # 728 * 19 * 19
        x3 = self.xception_rgb.model.fea_part1_3(x2 + y2)
        y3 = self.xception_srm.model.fea_part1_3(x2 + y2)
        y3 = self.lfe0(y3, x3)

        # 728 * 19 * 19
        x4 = self.xception_rgb.model.fea_part2_0(x3)
        y4 = self.xception_srm.model.fea_part2_0(y3)
        y4 = self.lfe1(y4, x4)

        # 728 * 19 * 19
        x5 = self.xception_rgb.model.fea_part2_1(x4)
        y5 = self.xception_srm.model.fea_part2_1(y4)
        y5 = self.lfe2(y5, x5)

        # 2048 * 10 * 10
        x6 = self.xception_rgb.model.fea_part3(x5)
        y6 = self.xception_srm.model.fea_part3(y5)

        x0u, x1u, x2u, x3u, x4u, x5u = self.score0(x0), self.score1(x1), self.score2(x2), self.score3(x3), self.score4(
            x4), self.score5(x5)
        x4m = self.msff(x4u, x5u)
        x3m = self.msff(x3u, x5u)
        x2m = self.msff(x2u, x5u)
        x1m = self.msff(x1u, x5u)
        x0m = self.msff(x0u, x5u)

        x5_reduced = self.score5(x5)
        seg_feas = torch.cat((x0m, x1m, x2m, x3m, x4m, x5_reduced), dim=1)

        y0m = self.pad_max_pool(y0)
        y1m = self.pad_max_pool(y1)
        y2m = self.pad_max_pool(y2)
        y3m = self.pad_max_pool(y3)
        y5m = self.pad_max_pool(y5)
        mul_feas = torch.cat((y0m, y1m, y2m, y3m, y5m), dim=1)
        cls_feas = self.HBFusion(mul_feas, y6)

        return cls_feas, seg_feas

    def forward(self, x, mask=None):
        # 检查是否提供了掩码
        has_mask = mask is not None

        # 如果没有提供掩码，创建一个全零掩码
        if not has_mask:
            mask = torch.zeros((x.size(0), 1, x.size(2), x.size(3)), device=x.device)

        feas = self.features(x)
        cls_preds = self.cls_header(feas[0])
        seg_preds = self.seg_header(feas[1])

        if mask is not None:
            if isinstance(mask, list):
                for i in range(len(mask)):
                    mask[i] = self.get_mask(mask[i])
                    mask[i][mask[i] > 0] = 1.0
            else:
                mask = self.get_mask(mask)
                mask[mask > 0] = 1.0

        # 如果没有提供掩码，返回时不包含掩码
        if not has_mask:
            return cls_preds, seg_preds
        else:
            return cls_preds, seg_preds, mask


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
