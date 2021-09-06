from os.path import join
import os
import cv2
import argparse
import numpy as np

from utils import bbox_iou, load_gt
from mosse import mosse


parse = argparse.ArgumentParser()
parse.add_argument('--lr', type=float, default=0.125, help='the learning rate')
parse.add_argument('--sigma', type=float, default=20, help='the sigma')
parse.add_argument('--num_pretrain', type=int, default=128, help='the number of pretrain')
parse.add_argument('--rotate', action='store_true', help='if rotate image during pre-training.')
parse.add_argument('--record', action='store_true', help='record the frames')
parse.add_argument('--visualize', action='store_true', default=False)
args = parse.parse_args()


SEQUENCE = 'Car4'

seqdir = join('datasets', SEQUENCE)
imgdir = join(seqdir, 'img')
imgnames = os.listdir(imgdir)
imgnames.sort()

tracker = mosse(args, seqdir)
results = tracker.track_sequence()

gt_boxes = load_gt(join('datasets', SEQUENCE, 'groundtruth_rect.txt'))

ious = []
for imgname, res_box, gt_box in zip(imgnames, results, gt_boxes):
    imgpath = join(imgdir, imgname)
    iou = bbox_iou(res_box, gt_box)
    ious.append(iou)

    if args.visualize:
        img = cv2.imread(imgpath)
        cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]), (0, 255, 0, 2))
        cv2.rectangle(img, (res_box[0], res_box[1]), (res_box[0]+res_box[2], res_box[1]+res_box[3]), (255, 0, 0, 2))
        cv2.imshow('gt', img)
        if cv2.waitKey(0) == ord('q'):
            break

print('Mean IoU:', np.mean(ious))