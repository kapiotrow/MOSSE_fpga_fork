# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import json
import sys
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d, ReLU

from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantReLU
from brevitas.core.restrict_val import RestrictValueType
# from .common import CommonWeightQuant, CommonActQuant
from brevitas_examples.bnn_pynq.models.common import CommonWeightQuant, CommonActQuant

# from models.mymodel import YOLOLayer


POOL_SIZE = 2
KERNEL_SIZE = 3
# FOR EXAMPLE CNV FROM FINN ONLY:
CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False


class YOLO_finn(Module):

    def __init__(self, config, in_bit_width=8, in_ch=3):
        super(YOLO_finn, self).__init__()

        self.anchors = config['anchors']
        self.backbone = CNVBackbone(config, in_bit_width=in_bit_width, in_ch=in_ch)
        mergetwo = config['merge_two_imgs'] if 'merge_two_imgs' in config else False
        self.yololayer = YOLOLayer(self.anchors, mergetwo=mergetwo)
        self.yolo_layers = [self.yololayer]


    def forward(self, x):

        img_size = x.shape[-2:]

        yolo_out, out = [], []

        x = self.backbone(x)
        # print('backbone features:', x.shape)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x  


class CNVBackbone(Module):

    def __init__(self, config, in_bit_width=8, in_ch=3):
        super(CNVBackbone, self).__init__()

        self.output_shape_printed = False
        self.conv_out_ch_pool = config['conv_out_ch_pool']
        self.w_a_bit_widths = config['w_a_bit_widths']
        self.batchnorms = config['batchnorms']
        self.conv_strides = config['conv_strides']
        self.conv_sizes = config['conv_sizes']
        self.conv_padding = config['conv_padding']
        self.use_relu = config['use_relu']
        assert len(self.conv_out_ch_pool) == len(self.w_a_bit_widths) == len(self.batchnorms) == len(self.conv_strides) == len(self.conv_sizes)
        self.conv_features = ModuleList()
        # self.linear_features = ModuleList()

        self.conv_features.append(QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        for (out_ch, is_pool_enabled), (weight_bit_width, act_bit_width), is_batchnorm_enabled, conv_stride, conv_size in zip(self.conv_out_ch_pool,
                                                                                                                            self.w_a_bit_widths,
                                                                                                                            self.batchnorms,
                                                                                                                            self.conv_strides,
                                                                                                                            self.conv_sizes):
            # print(out_ch, is_pool_enabled, weight_bit_width, act_bit_width)
            self.conv_features.append(QuantConv2d(
                kernel_size=conv_size,
                in_channels=in_ch,
                out_channels=out_ch,
                padding=self.conv_padding,
                stride=conv_stride,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            in_ch = out_ch

            if is_batchnorm_enabled:
                self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))

            if self.use_relu:
                self.conv_features.append(QuantReLU())

            if act_bit_width != 0:
                self.conv_features.append(QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bit_width))

            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))
        
        for m in self.modules():
          if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)


    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)


    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
            # print('SHAPE:', x.shape)
        if not self.output_shape_printed:
            self.output_shape_printed = True   
            # print('SHAPE:', x.shape)
        return x


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    # print('nx, ny:', nx, ny)
    self.img_size = max(img_size)
    self.anchors = self.anchors.to(device)
    # print('anchors device:', self.anchors.device)
    # self.stride = self.img_size / max(ng)
    self.stride_x = img_size[1] / nx
    self.stride_y = img_size[0] / ny
    # print('strides x y :', self.stride_x, self.stride_y)
    # print('stride:', self.stride)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))
    # print('gridxy shape:', self.grid_xy[0, 0, :, :, 1])

    # build wh gains
    # print('self anchors:', self.anchors)
    self.anchor_vec = self.anchors.float() # list of [h, w]
    self.anchor_vec[:, 0] /= self.stride_y
    self.anchor_vec[:, 1] /= self.stride_x
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny

class YOLOLayer(Module):
    def __init__(self, anchors, mergetwo=False):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (6)
        self.no = 6  # number of outputs
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
        self.mergetwo = mergetwo

    def forward(self, p, img_size):

        # print('input p in yololayer:', p.shape)
        
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # shape: (bs, na, ny, nx, output) and shifts it by meshgrid
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # anchor_wh: anchor_size/network_stride
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            
            # transform boxes from last tensor space to input tensor space
            # io[..., :4] *= self.stride  # 原始像素尺度
            io[..., 0] *= self.stride_x
            io[..., 1] *= self.stride_y
            io[..., 2] *= self.stride_x
            io[..., 3] *= self.stride_y

            # sigmoid for classification
            torch.sigmoid_(io[..., 4:])
            
            if self.mergetwo:
                top_io = io[:, :, :ny//2, :, :]
                top_io = top_io.reshape(bs, -1, self.no)
                bottom_io = io[:, :, ny//2:, :, :]
                bottom_io = bottom_io.reshape(bs, -1, self.no)
                result = torch.cat((top_io, bottom_io), dim=1)
            else:
                result = io.view(bs, -1, self.no)

            return result, p


class CNVmy(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super(CNVmy, self).__init__()

        self.conv_features = ModuleList()
        # self.linear_features = ModuleList()

        self.conv_features.append(QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        for i, (out_ch, is_pool_enabled) in enumerate(CNV_OUT_CH_POOL):
            self.conv_features.append(QuantConv2d(
                kernel_size=KERNEL_SIZE,
                in_channels=in_ch,
                out_channels=out_ch,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            if i != len(CNV_OUT_CH_POOL) - 1:
                self.conv_features.append(QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bit_width))
            else:
                print('USUNIETE QUANT')
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))

        # for in_features, out_features in INTERMEDIATE_FC_FEATURES:
        #     self.linear_features.append(QuantLinear(
        #         in_features=in_features,
        #         out_features=out_features,
        #         bias=False,
        #         weight_quant=CommonWeightQuant,
        #         weight_bit_width=weight_bit_width))
        #     self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
        #     self.linear_features.append(QuantIdentity(
        #         act_quant=CommonActQuant,
        #         bit_width=act_bit_width))

        # self.linear_features.append(QuantLinear(
        #     in_features=LAST_FC_IN_FEATURES,
        #     out_features=num_classes,
        #     bias=False,
        #     weight_quant=CommonWeightQuant,
        #     weight_bit_width=weight_bit_width))
        # self.linear_features.append(TensorNorm())
        
        for m in self.modules():
          if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)


    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        # for mod in self.linear_features:
        #     if isinstance(mod, QuantLinear):
        #         mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            print('SHAPE:', x.shape)
            x = mod(x)
        # x = x.view(x.shape[0], -1)
        # for mod in self.linear_features:
        #     x = mod(x)
        return x


def get_CNVmy(weights_path, convonly=False):

    model = CNVmy(num_classes=10, weight_bit_width=1, act_bit_width=1, in_bit_width=8, in_ch=3)
    state_dict = torch.load(weights_path)

    if convonly:
        new_state_dict = {}
        for k in state_dict.keys():
            if 'linear_features' not in k:
                new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()

    print(model.conv_features)

    return model


class CNV(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super(CNV, self).__init__()

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(QuantConv2d(
                kernel_size=KERNEL_SIZE,
                in_channels=in_ch,
                out_channels=out_ch,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=act_bit_width))

        self.linear_features.append(QuantLinear(
            in_features=LAST_FC_IN_FEATURES,
            out_features=num_classes,
            bias=False,
            weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())
        
        for m in self.modules():
          if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)


    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x


def cnv(cfg):
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')
    net = CNV(weight_bit_width=weight_bit_width,
              act_bit_width=act_bit_width,
              in_bit_width=in_bit_width,
              num_classes=num_classes,
              in_ch=in_channels)
    return net


class TensorNorm(Module):
    def __init__(self, eps=1e-4, momentum=0.1):
        super().__init__()

        self.eps = eps
        self.momentum = momentum
        self.weight = torch.nn.Parameter(torch.rand(1))
        self.bias = torch.nn.Parameter(torch.rand(1))
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            mean = x.mean()
            unbias_var = x.var(unbiased=True)
            biased_var = x.var(unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
            inv_std = 1 / (biased_var + self.eps).pow(0.5)
            return (x - mean) * inv_std * self.weight + self.bias
        else:
            return ((x - self.running_mean) / (self.running_var + self.eps).pow(0.5)) * self.weight + self.bias


def get_pretrained_backbone(config_path, weights_path):

    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    yolo_model = YOLO_finn(config).to(torch.device('cpu'))
    print(yolo_model.backbone.conv_features[1].weight[0])
    checkpoint_dict = torch.load(weights_path, map_location='cpu')
    yolo_model.load_state_dict(checkpoint_dict['model'])
    print(yolo_model.backbone.conv_features[1].weight[0])
    the_backbone = yolo_model.backbone
    the_backbone.eval()
    print(the_backbone.conv_features)

    return the_backbone