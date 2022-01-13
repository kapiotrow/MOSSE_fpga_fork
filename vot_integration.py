from os.path import join
import os
import cv2
import argparse
import numpy as np
import vot
import sys

from deep_mosse import DeepMosse


parse = argparse.ArgumentParser()
parse.add_argument('--lr', type=float, default=0.236111, help='the learning rate')
parse.add_argument('--sigma', type=float, default=18, help='the sigma')
parse.add_argument('--lambd', type=float, default=0.01, help='regularization parameter')
parse.add_argument('--num_pretrain', type=int, default=0, help='the number of pretrain')
parse.add_argument('--rotate', action='store_true', help='if rotate image during pre-training.')
parse.add_argument('--debug', action='store_true', default=False)
parse.add_argument('--deep', action='store_true', default=True, help='whether to use deep features instead of grayscale')
args = parse.parse_args()

handle = vot.VOT("rectangle")
selection = handle.region()
selection = [selection.x, selection.y, selection.width, selection.height]

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
tracker = DeepMosse(image, selection, args)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    region = tracker.track(image)
    region = vot.Rectangle(region[0], region[1], region[2], region[3])
    handle.report(region)