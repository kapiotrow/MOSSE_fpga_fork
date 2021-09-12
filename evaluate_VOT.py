from os.path import join
import os
import cv2
import argparse
import numpy as np

from utils import bbox_iou, load_gt
from mosse import mosse, mosse_old


def show_VOT_dataset(dataset_path):

    sequences = os.listdir(dataset_path)
    for seq in sequences:
        seqpath = join(dataset_path, seq)
        imgpath = join(seqpath, 'img')
        gt = load_gt(join(seqpath, 'groundtruth.txt'))
        imgnames = os.listdir(imgpath)
        imgnames.sort()
        for imgname, bbox in zip(imgnames, gt):
            img = cv2.imread(join(imgpath, imgname))
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
            cv2.imshow('demo', img)
            if cv2.waitKey(0) == ord('q'):
                break


parse = argparse.ArgumentParser()
parse.add_argument('--lr', type=float, default=0.125, help='the learning rate')
parse.add_argument('--sigma', type=float, default=10, help='the sigma')
parse.add_argument('--num_pretrain', type=int, default=128, help='the number of pretrain')
parse.add_argument('--rotate', action='store_true', help='if rotate image during pre-training.')
parse.add_argument('--record', action='store_true', help='record the frames')
parse.add_argument('--visualize', action='store_true', default=False)
parse.add_argument('--seq', type=str, default='all')
args = parse.parse_args()


# show_VOT_dataset('../datasets/VOT2013')

DATASET_DIR = '../datasets/VOT2013'
if args.seq == 'all':
    sequences = os.listdir(DATASET_DIR)
else:
    sequences = [args.seq]

ious_per_sequence = {}
for sequence in sequences:
    print('Testing', sequence)
    seqdir = join(DATASET_DIR, sequence)
    imgdir = join(seqdir, 'img')
    imgnames = os.listdir(imgdir)                  
    imgnames.sort()

    # tracker = mosse(args, seqdir, FFT_SIZE=200)
    tracker = mosse_old(args, seqdir)
    results = tracker.start_tracking()

    gt_boxes = load_gt(join(seqdir, 'groundtruth.txt'))
    # print('results, gt:', len(results), len(gt_boxes))

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
            # print(iou)
            if cv2.waitKey(0) == ord('q'):
                break

    ious_per_sequence[sequence] = np.mean(ious)

for k, v in ious_per_sequence.items():
    print(k, v)

print('Mean IoU over dataset:', np.mean(list(ious_per_sequence.values())) )