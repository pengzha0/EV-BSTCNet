import torch
import torch.nn as nn
import math
from torch.cuda.amp import autocast
import torch.nn.functional as F
from model.tcn import MultibranchTemporalConvNet

def conv1x3x3(in_planes, out_planes, stride=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=(1, stride, stride), bias=False)





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se

        if self.se:
            self.gap = nn.AdaptiveAvgPool3d(1)
            self.conv3 = conv1x1x1(planes, planes//16)
            self.conv4 = conv1x1x1(planes//16, planes)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # print(out.shape)# torch.Size([16, 16, 210, 22, 22])



        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        if self.se:
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()

            out = out * w

        out = out + residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, block, layers, se=False, **kwargs):
        super(ResNet18, self).__init__()
        in_channels = kwargs['in_channels']

        # print(self.low_rate)

        self.base_channel = kwargs['base_channel']
        # self.inplanes = (self.base_channel + self.base_channel//self.alpha*self.t2s_mul) if self.low_rate else self.base_channel // self.alpha
        self.inplanes = 64
        # print(self.inplanes)
        self.conv1 = nn.Conv3d(in_channels, self.base_channel,
                               kernel_size=(5, 7, 7),
                               stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.base_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.se = se
        self.layer1 = self._make_layer(block, self.base_channel, layers[0])
        self.layer2 = self._make_layer(block, 2 * self.base_channel, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * self.base_channel, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * self.base_channel, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)


        self.bn2 = nn.BatchNorm1d(8 * self.base_channel)
        # if self.low_rate:
        #     self.bn2 = nn.BatchNorm1d(8*self.base_channel + 8*self.base_channel//self.alpha*self.t2s_mul)
        # elif self.t2s_mul == 0:
        #     self.bn2 = nn.BatchNorm1d(16*self.base_channel//self.alpha)
        self.init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        # print(self.inplanes)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        # print(planes)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        # self.inplanes += self.low_rate * block.expansion * planes // self.alpha * self.t2s_mul

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)# torch.Size([32, 64, 30, 22, 22])
        # print(laterals[0].shape)# torch.Size([32, 32, 30, 22, 22])
        # x = self.mfm1(laterals[0], x)
        # print(x.shape)# torch.Size([32, 64, 30, 22, 22])
        # x = torch.cat([x, laterals[0]], dim=1)
        # print(x.shape)# torch.Size([32, 96, 30, 22, 22])
        x = self.layer1(x)  # (b, 64, 30, 22, 22)

        # x = self.mfm2(laterals[1], x)
        # x = torch.cat([x, laterals[1]], dim=1) # (b, 80, 30, 22, 22)
        x = self.layer2(x)  # (b, 128, 30, 11, 11)

        # x = self.mfm3(laterals[2], x)
        # x = torch.cat([x, laterals[2]], dim=1) # (b, 160, 30, 11, 11)
        x = self.layer3(x)  # (2, 256, 30, 6, 6)

        # x = self.mfm4(laterals[3], x)
        # x = torch.cat([x, laterals[3]], dim=1) # (b, 320, 30, 6, 6)
        x = self.layer4(x)  # (b, 512, 30, 3, 3)

        # x = torch.cat([x, laterals[4]], dim=1) # (b, 640, 30, 3, 3)
        x = self.avgpool(x)  # (b, 640, 30, 1, 1)

        x = x.transpose(1, 2).contiguous()  # (b, 30, 640, 1, 1)
        x = x.view(-1, x.size(2))  # (b*30, 640)
        x = self.bn2(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


#
#
# class LowRateBranch(ResNet18):
#     def __init__(self, block, layers, se, n_frame=30, **kargs):
#         super().__init__(block, layers, se, n_frame=n_frame, **kargs)
#         self.base_channel = kargs['base_channel']
#         self.init_params()
#
#     def forward(self, x):
#         x = x.transpose(1, 2)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         # print(x.shape)# torch.Size([32, 64, 30, 22, 22])
#         # print(laterals[0].shape)# torch.Size([32, 32, 30, 22, 22])
#         # x = self.mfm1(laterals[0], x)
#         # print(x.shape)# torch.Size([32, 64, 30, 22, 22])
#         # x = torch.cat([x, laterals[0]], dim=1)
#         # print(x.shape)# torch.Size([32, 96, 30, 22, 22])
#         x = self.layer1(x) # (b, 64, 30, 22, 22)
#
#         # x = self.mfm2(laterals[1], x)
#         # x = torch.cat([x, laterals[1]], dim=1) # (b, 80, 30, 22, 22)
#         x = self.layer2(x) # (b, 128, 30, 11, 11)
#
#         # x = self.mfm3(laterals[2], x)
#         # x = torch.cat([x, laterals[2]], dim=1) # (b, 160, 30, 11, 11)
#         x = self.layer3(x) # (2, 256, 30, 6, 6)
#
#         # x = self.mfm4(laterals[3], x)
#         # x = torch.cat([x, laterals[3]], dim=1) # (b, 320, 30, 6, 6)
#         x = self.layer4(x) # (b, 512, 30, 3, 3)
#
#         # x = torch.cat([x, laterals[4]], dim=1) # (b, 640, 30, 3, 3)
#         x = self.avgpool(x) # (b, 640, 30, 1, 1)
#
#         x = x.transpose(1, 2).contiguous() # (b, 30, 640, 1, 1)
#         x = x.view(-1, x.size(2)) # (b*30, 640)
#         x = self.bn2(x)
#
#         return x
#

class MultiBranchNet(nn.Module):
    def __init__(self, args):
        super(MultiBranchNet, self).__init__()
        self.args = args
        self.low_rate_branch = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], se=args.se, in_channels=1,
                                            low_rate=1, base_channel=args.base_channel)



    def forward(self, x):
        b = x.size(0)
        # y, laterals = self.high_rate_branch(y)
        x = self.low_rate_branch(x)

        # x = x.view(b, -1, 8*self.args.base_channel+8*self.args.base_channel//self.args.alpha*self.args.t2s_mul)
        x = x.view(b, -1, 8 * self.args.base_channel)
        return x


class EV_STCNet(nn.Module):
    def __init__(self, args):
        super(EV_STCNet, self).__init__()
        self.args = args
        self.mbranch = MultiBranchNet(args)
        self.add_channel = 1 if self.args.word_boundary else 0

        if self.args.back_type == 'TCN':
            tcn_options = {'num_layers': args.num_layers,
                           'kernel_size': [3,5,7],
                           'dropout': 0.2,
                           'dwpw': False,
                           'width_mult': 1
                           }
            self.TCN = MultibranchTemporalConvNet(num_inputs=512 + self.add_channel, num_channels=[768, 768, 768, 768 , 768,768,768], tcn_options=tcn_options)
            self.v_cls = nn.Linear(768, self.args.n_class)
        elif self.args.back_type == 'GRU':
            self.gru = nn.GRU(512 + self.add_channel, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
            self.v_cls = nn.Linear(1024 * 2, self.args.n_class)

        self.dropout = nn.Dropout(p=0.5)
        # args.base_channel = 64, self.args.alpha = 4, self.args.t2s_mul = 2
        # in_dim = 8 * args.base_channel + 8 * args.base_channel // self.args.alpha * self.args.t2s_mul
        # in_dim = 8 * args.base_channel


    def forward(self, data):
        event_low = data['event_low']
        if self.training:
            with autocast():
                feat = self.mbranch(event_low)
                feat = self.dropout(feat)
                feat = feat.float()
        else:
            feat = self.mbranch(event_low)
            feat = feat.float()


        # print(data['word_boundary_low'].shape, feat.shape)
        # torch.Size([32, 60]) torch.Size([32, 60, 512])
        if self.args.word_boundary:
            feat = torch.cat([feat, data['word_boundary_low'].unsqueeze(-1)], -1)

        if self.args.back_type == 'TCN':
            feat = feat.permute(0, 2, 1)
            logit = self.v_cls(self.TCN(feat).mean(-1))
            return logit

        elif self.args.back_type == 'GRU':
            self.gru.flatten_parameters()

            feat, _ = self.gru(feat)
            logit = self.v_cls(self.dropout(feat)).mean(1)

            return logit

