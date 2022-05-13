from os.path import join
import os
import cv2
import argparse
import numpy as np
import vot
import sys

from deep_mosse_FOREVAL import DeepMosse


parse = argparse.ArgumentParser()
parse.add_argument('--lr', type=float, default=0.025, help='the learning rate')
parse.add_argument('--sigma', type=float, default=17, help='the sigma')
parse.add_argument('--lambd', type=float, default=0.01, help='regularization parameter')
parse.add_argument('--num_pretrain', type=int, default=0, help='the number of pretrain')
parse.add_argument('--rotate', action='store_true', default=False, help='if rotate image during pre-training.')
parse.add_argument('--debug', action='store_true', default=False)
parse.add_argument('--deep', action='store_true', default=True, help='whether to use deep features instead of grayscale')
parse.add_argument('--search_region_scale', type=float, default=2)
parse.add_argument('--clip_search_region', action='store_true', help='whether to clip search region to image borders or make a border (zero-pad or reflect)')
parse.add_argument('--scale_factor', type=float, default=1.005)
parse.add_argument('--num_scales', type=int, default=5)
parse.add_argument('--border_type', type=str, default='reflect', help='reflect or constant')
parse.add_argument('--quantized', default=True, action='store_true')
args = parse.parse_args()
print('args:', args)

handle = vot.VOT("rectangle")
selection = handle.region()
selection = [selection.x, selection.y, selection.width, selection.height]

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
tracker = DeepMosse(image, selection, args, FFT_SIZE=224, buffer_features=True)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    region = tracker.track(image)
    region = vot.Rectangle(region[0], region[1], region[2], region[3])
    handle.report(region)