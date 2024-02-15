from os.path import join
import os
import cv2
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import json

from utils import bbox_iou, load_gt, init_seeds
from mosse import mosse
from deep_mosse import DeepMosse


# init_seeds()


def show_VOT_dataset(dataset_path, sequences=None, draw_trajectory=False, standard='vot2015', mosaic=False):

    if sequences == None:
        sequences = os.listdir(dataset_path)
        sequences = [seq for seq in sequences if os.path.isdir(join(dataset_path, seq))]
    # sequences = ['david', 'iceskater', 'face', 'singer', 'bicycle', 'bolt']

    imgs_per_seq = 6
    mosaic_type = 'rowwise'
    if mosaic_type == 'rowwise':
        f, axarr = plt.subplots(len(sequences), imgs_per_seq)
    else:
        f, axarr = plt.subplots(imgs_per_seq, len(sequences))

    for i_seq, seq in enumerate(sequences):
        seqpath = join(dataset_path, seq)
        imgpath = join(seqpath, 'color')
        gt = load_gt(join(seqpath, 'groundtruth.txt'), standard=standard)
        imgnames = os.listdir(imgpath)
        imgnames.sort()
        trajectory = []

        if mosaic:
            
            img_indices = list(np.linspace(0, len(imgnames)-1, imgs_per_seq))
            img_indices = [int(el) for el in img_indices]
            # print('indices:', img_indices)
            # print('lens:', len(gt), len(imgnames))
            for i, img_ind in enumerate(img_indices):
                bbox = gt[img_ind]
                img = cv2.imread(join(imgpath, imgnames[img_ind]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
                if mosaic_type == 'rowwise':
                    axarr[i_seq, i].imshow(img)
                    axarr[i_seq, i].axis('off')
                else:
                    axarr[i, i_seq].imshow(img)
                    axarr[i, i_seq].axis('off')
        else:
            for imgname, bbox in zip(imgnames, gt):
                img = cv2.imread(join(imgpath, imgname))
                if draw_trajectory:
                    center_xy = (bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2)
                    trajectory.append(center_xy)
                    for i, center in enumerate(trajectory):
                        if i >= 2:
                            cv2.line(img, trajectory[i-1], trajectory[i], color=(0, 255, 255), thickness=2)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
                cv2.circle(img, (bbox[0], bbox[1]), radius=2, color=(0, 0, 255), thickness=-1)
                cv2.circle(img, (bbox[0]+bbox[2], bbox[1]+bbox[3]), radius=2, color=(0, 255, 0), thickness=-1)
                cv2.imshow(seq, img)
                print('Press anything for next frame, q to quit or s to skip sequence')
                key = cv2.waitKey(0)
                if key == ord('q') or key == ord('s'):
                    break
            if key == ord('q'):
                break
    if mosaic:
        plt.axis('off')
        plt.show()


def test_sequence(sequence, write_images=False):

    seqdir = join(DATASET_DIR, sequence)
    imgdir = join(seqdir, 'color')
    imgnames = os.listdir(imgdir)                  
    imgnames.sort()

    print('init frame:', join(imgdir, imgnames[0]))
    init_img = cv2.imread(join(imgdir, imgnames[0]))
    gt_boxes = load_gt(join(seqdir, 'groundtruth.txt'), standard='vot2015')
    tracker = DeepMosse(init_img, gt_boxes[0], config=config, debug=args.debug)
    if args.debug:
        position = gt_boxes[0]
        cv2.rectangle(init_img, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2)
        cv2.imshow('demo', init_img)
        cv2.waitKey(0)


    results = []
    for imgname in imgnames[1:]:
        img = cv2.imread(join(imgdir, imgname))
        frame_num = int(imgname.split('.')[0])
        print(imgname)

        # start = time.time()
        position = tracker.track(img)
        # print('fps:', 1/(time.time() - start))

        results.append(position.copy())

        if args.debug:
            position = [round(x) for x in position]
            cv2.rectangle(img, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2)
            cv2.imshow('demo', img)
            if cv2.waitKey(0) == ord('q'):
                break
        
        if write_images:
            position = [round(x) for x in position]
            cv2.rectangle(img, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2)
            cv2.imwrite('./output/images/{:03d}.png'.format(tracker.current_frame), img)

    return results, gt_boxes


parse = argparse.ArgumentParser()
parse.add_argument('--config', type=str, default='configs/config.json')
# parse.add_argument('--sequences', type=str, help='path to a directory with sequences')
parse.add_argument('--seq', type=str, default='all')
parse.add_argument('--params_search', action='store_true', default=False)
parse.add_argument('--debug', action='store_true', default=False)
parse.add_argument('--write', action='store_true', default=False)
args = parse.parse_args()

with open(args.config, 'r') as json_file:
    config = json.load(json_file)

# DATASET_DIR = '../datasets/VOT2013'
DATASET_DIR = 'workspace_vot2015/sequences'



if args.seq == 'all':
    sequences = os.listdir(DATASET_DIR)
    sequences = [seq for seq in sequences if os.path.isdir(join(DATASET_DIR, seq))]
else:
    sequences = [args.seq]

print(sequences)
show_VOT_dataset(DATASET_DIR, sequences=sequences, draw_trajectory=False, standard='vot2015', mosaic=False)

best_score = 0
best_params = []

if args.params_search: 

    for sigma in range(1, 20):
        for lr in list(np.linspace(0.025, 0.5, 10)):
            config.sigma = sigma
            config.lr = lr
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
    ious_per_sequence = {}
    for sequence in sequences:
        # print('Testing', sequence)
        results, gt_boxes = test_sequence(sequence, write_images=args.write)

        ious = []
        for res_box, gt_box in zip(results, gt_boxes[1:]):
            iou = bbox_iou(res_box, gt_box)
            ious.append(iou)

        ious_per_sequence[sequence] = np.mean(ious)
        print(sequence, ':', np.mean(ious))

    # for k, v in ious_per_sequence.items():
    #     print(k, v)
    print('Mean IoU:', np.mean(list(ious_per_sequence.values())))

