import numpy as np
import cv2
import imutils
import torch
from torch.nn import Module
import torchvision.models as models
import json
import matplotlib.pyplot as plt

from finnmodels import YOLO_finn


def init_seeds(seed=0):
    np.random.seed(seed)


def get_VGG_backbone(pretrained=True):

    vgg = models.vgg11(pretrained=pretrained, progress=True)
    vgg.eval()

    return vgg.features[:3]


def get_CF_backbone(config_path, weights_path):

    class CFBackbone(Module):

        def __init__(self, conv_features, take_first_n):
            super(CFBackbone, self).__init__()
            self.conv_features = conv_features[:take_first_n]

        def forward(self, x):
            x = 2.0 * x - torch.tensor([1.0], device=x.device)
            for mod in self.conv_features:
                x = mod(x)
            
            return x


    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    detector = YOLO_finn(config).to(torch.device('cpu'))
    checkpoint_dict = torch.load(weights_path, map_location='cpu')
    detector.load_state_dict(checkpoint_dict['model'])

    backbone = detector.backbone
    cf_backbone = CFBackbone(backbone.conv_features, 5)
    cf_backbone.eval()

    return cf_backbone


# pre-processing the image... DEPRECATED
def pre_process(img):
    # print('USING DEPRECATED PREPROCESSING')
    # get the size of the img...
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # use the hanning window...
    window = window_func_2d(height, width)
    img = img * window

    return img


def load_gt(gt_file):

    with open(gt_file, 'r') as file:
        lines = file.readlines()

    delimiters = [',', '\t']

    for d in delimiters:
        if d in lines[0]:
            lines = [line.split(d) for line in lines]
            break
    lines = [[int(float(coord)) for coord in line] for line in lines]

    return lines


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # print('boxes shapes:', box1.shape, box2.shape)
 
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
    b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
    b2_x1, b2_x2 = box2[0], box2[0] + box2[2]
    b2_y1, b2_y2 = box2[1], box2[1] + box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, a_min=0, a_max=None) * np.clip(inter_rect_y2 - inter_rect_y1, a_min=0, a_max=None)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def pad_img(img, padded_size, pad_type='center'):

    if padded_size == 0:
        return img, [0, 0, 0, 0]

    height, width = img.shape
    # print('hw:', height, width)
    assert height <= padded_size and width <= padded_size

    if pad_type == 'topleft':
        result = np.zeros((padded_size, padded_size))
        result[:height, :width] = img
        padding = [0, padded_size-height, 0, padded_size-width]

    elif pad_type == 'center':
        h_diff = padded_size - height
        w_diff = padded_size - width
        if h_diff % 2 == 0:
            top_pad = h_diff / 2
            bottom_pad = top_pad
        else:
            top_pad = h_diff // 2
            bottom_pad = top_pad + 1
        
        if w_diff % 2 == 0:
            left_pad = w_diff / 2
            right_pad = left_pad
        else:
            left_pad = w_diff // 2
            right_pad = left_pad + 1

        padding = [top_pad, bottom_pad, left_pad, right_pad]
        padding = [int(pad) for pad in padding]
        top_pad, bottom_pad, left_pad, right_pad = padding
        # print('padding:', padding)
        result = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT)
    
    # cv2.imshow('padded', result.astype(np.uint8))
    # print(result)

    return result, padding


# used for linear mapping...
def linear_mapping(img):
    return (img - img.min()) / (img.max() - img.min())



def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win

# img in [C, H, W] shape
def random_warp(img, i='0'):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()

    channels, height, width = img.shape
    img = img.transpose(1, 2, 0)
    img_rot = imutils.rotate_bound(img, r)
    img_resized = cv2.resize(img_rot, (width, height))
    # cv2.imshow(i+' sample', img_resized)
    if channels == 1:
        img_resized = np.expand_dims(img_resized, axis=0)
    # print('shape:', img_resized.shape)
    # rotate the image...
    # matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    # img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    # img_rot = img_rot.astype(np.float32) / 255
    return img_resized


