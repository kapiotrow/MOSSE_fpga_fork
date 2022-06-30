import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas_examples.bnn_pynq.models.common import CommonWeightQuant, CommonActQuant
from brevitas.core.restrict_val import RestrictValueType


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class FINNLayer(nn.Module):
    
    def __init__(self, w_bit_width, a_bit_width, channels=64, include_pooling=False):
        super(FINNLayer, self).__init__()

        self.conv_features = nn.ModuleList()
        self.conv_features.append(QuantIdentity( 
                                    act_quant=CommonActQuant,
                                    bit_width=a_bit_width
                                    # min_val=- 1.0,
                                    # max_val=1.0 - 2.0 ** (-7),
                                    # narrow_range=False,
                                    # restrict_scaling_type=RestrictValueType.POWER_OF_TWO
                                    ))

        self.conv_features.append(QuantConv2d(
                                    kernel_size=3,
                                    in_channels=3,
                                    out_channels=channels,
                                    padding=1,
                                    stride=1,
                                    bias=True,
                                    weight_quant=CommonWeightQuant,
                                    weight_bit_width=w_bit_width))
        self.conv_features.append(QuantReLU(inplace=True))
        # self.conv_features.append(nn.ReLU(inplace=True))
        if include_pooling:
            self.conv_features.append(nn.MaxPool2d(2))

    def forward(self, x):
        # x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            # if isinstance(mod, type(nn.ReLU())) or isinstance(mod, type(QuantReLU())):
            #     print('pre:')
            #     print(x)
            x = mod(x)
            # if isinstance(mod, type(nn.ReLU())) or isinstance(mod, type(QuantReLU())):
            #     print('post:')
            #     print(x)

        return x


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    a_bit_width = 32
    w_bit_width = 32
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            if i == 0:
                layers += [FINNLayer(8, 8)]
                # layers += [QuantIdentity( 
                #             act_quant=CommonActQuant,
                #             bit_width=a_bit_width)]
                # layers += [QuantConv2d(
                #             kernel_size=3,
                #             in_channels=in_channels,
                #             out_channels=64,
                #             padding=1,
                #             stride=1,
                #             bias=True,
                #             weight_quant=CommonWeightQuant,
                #             weight_bit_width=w_bit_width)]
                               
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        pretrained_state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model_state_dict = model.state_dict()

        state_dict = {}
        for (pretrained_key, v), model_key in zip(pretrained_state_dict.items(), model_state_dict.keys()):
            # print(pretrained_key, model_key)
            if 'features' in pretrained_key:
                state_dict[model_key] = v
            else:
                state_dict[pretrained_key] = v

        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def kn_init_weights(teacher, student):
    
    teacher_dict = teacher.state_dict()
    student_dict = student.state_dict()

    state_dict = {}
    for (pretrained_key, v), model_key in zip(teacher_dict.items(), student_dict.keys()):
        print(pretrained_key, model_key)
        state_dict[model_key] = v

    student.load_state_dict(state_dict)


# loads finnlayer weights to pretrained vgg with finnlayer
def load_finnlayer(model, checkpoint_path):

    model_dict = model.state_dict()
    finnlayer_dict = torch.load(checkpoint_path, map_location='cuda:0')['state_dict']
    # print(finnlayer_dict.keys())
    target_keys = ['conv_features.1.weight', 'conv_features.1.bias']
    for k, v in model_dict.items():
        for target_key in target_keys:
            if target_key in k:
                # print(k)
                model_dict[k] = finnlayer_dict['module.' + target_key]

    model.load_state_dict(model_dict)


# assuming weights are for 64 channels
def get_finnlayer(weights_path, w_a_bitwidths=(4, 4), channels=64, load_weights=True, strict=True):

    model = FINNLayer(w_a_bitwidths[0], w_a_bitwidths[1], channels, include_pooling=True)
    weights_dict = torch.load(weights_path, map_location='cpu')['state_dict']

    new_dict = {}
    for k, v in weights_dict.items():
        if 'module.' in k:
            k = k.strip('module.')
            new_dict[k] = v[:channels]

    if load_weights:
        model.load_state_dict(new_dict, strict=strict)
    model.eval()

    return model