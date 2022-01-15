from os.path import join
import os
import cv2
import argparse
import numpy as np

from utils import bbox_iou, load_gt, init_seeds
from mosse import mosse
from deep_mosse import DeepMosse


# init_seeds()


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


def test_sequence(sequence):

    seqdir = join(DATASET_DIR, sequence)
    imgdir = join(seqdir, 'img')
    imgnames = os.listdir(imgdir)                  
    imgnames.sort()

    init_img = cv2.imread(join(imgdir, imgnames[0]))
    gt_boxes = load_gt(join(seqdir, 'groundtruth.txt'))
    tracker = DeepMosse(init_img, gt_boxes[0], args)

    if args.debug:
        position = gt_boxes[0]
        cv2.rectangle(init_img, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2)
        cv2.imshow('demo', init_img)
        cv2.waitKey(0)

    results = []
    for imgname in imgnames[1:]:
        img = cv2.imread(join(imgdir, imgname))
        position = tracker.track(img)
        results.append(position.copy())

        if args.debug:
            cv2.rectangle(img, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2)
            cv2.imshow('demo', img)
            if cv2.waitKey(0) == ord('q'):
                break

    return results, gt_boxes


parse = argparse.ArgumentParser()
parse.add_argument('--lr', type=float, default=0.236111, help='the learning rate')
parse.add_argument('--sigma', type=float, default=18, help='the sigma')
parse.add_argument('--lambd', type=float, default=0.01, help='regularization parameter')
parse.add_argument('--num_pretrain', type=int, default=0, help='the number of pretrain')
parse.add_argument('--rotate', action='store_true', help='if rotate image during pre-training.')
parse.add_argument('--seq', type=str, default='all')
parse.add_argument('--params_search', action='store_true', default=False)
parse.add_argument('--debug', action='store_true', default=False)
parse.add_argument('--deep', action='store_true', default=False, help='whether to use deep features instead of grayscale')
parse.add_argument('--search_region_scale', type=int, default=2)
parse.add_argument('--clip_search_region', action='store_true', help='whether to clip search region to image borders or make a border (zero-pad or reflect)')
parse.add_argument('--scale_factor', type=float, default=1.005)
parse.add_argument('--num_scales', type=int, default=5)
parse.add_argument('--border_type', type=str, default='constant', help='reflect or constant')
args = parse.parse_args()
print('args:', args)
# show_VOT_dataset('../datasets/VOT2013')

DATASET_DIR = '../datasets/VOT2013'
CONV_NETWORK_NAME = 'yolo_finn_6conv_8w8a_160x320_5anchors'
NET_CONFIG = join('networks', CONV_NETWORK_NAME, 'config.json')
NET_WEIGHTS = join('networks', CONV_NETWORK_NAME, 'test_best.pt')

if args.seq == 'all':
    sequences = os.listdir(DATASET_DIR)
else:
    sequences = [args.seq]

print(sequences)

best_score = 0
best_params = []

if args.params_search: 

    for sigma in range(9, 20):
        for lr in list(np.linspace(0.025, 0.5, 10)):
            args.sigma = sigma
            args.lr = lr
            ious_per_sequence = {}
            for sequence in sequences:
                results, gt_boxes = test_sequence(sequence)

                ious = []
                for res_box, gt_box in zip(results, gt_boxes):
                    iou = bbox_iou(res_box, gt_box)
                    ious.append(iou)

                ious_per_sequence[sequence] = np.mean(ious)

            # for k, v in ious_per_sequence.items():
            #     print(k, v)
            score = np.mean(list(ious_per_sequence.values()))
            if score > best_score:
                best_score = score
                best_params = [sigma, lr]
            print('[{:.3f}, {:.3f}]: {:.3f}\tbest: {:.3f}'.format(sigma, lr, score, best_score))

    print('Finished. Best score:', best_score, 'best params:', best_params)

else:
    print('sigma: {:.3f}\tlr: {:.3f}'.format(args.sigma, args.lr))
    ious_per_sequence = {}
    for sequence in sequences:
        # print('Testing', sequence)
        results, gt_boxes = test_sequence(sequence)

        ious = []
        for res_box, gt_box in zip(results, gt_boxes[1:]):
            iou = bbox_iou(res_box, gt_box)
            ious.append(iou)

        ious_per_sequence[sequence] = np.mean(ious)
        print(sequence, ':', np.mean(ious))

    # for k, v in ious_per_sequence.items():
    #     print(k, v)
    print('Mean IoU:', np.mean(list(ious_per_sequence.values())))

