import numpy as np
import cv2
import os
from os.path import join


from utils import linear_mapping, pre_process, random_warp, pad_img, load_gt


FFT_SIZE = 0


"""
This module implements the basic correlation filter based tracking algorithm -- MOSSE

Date: 2018-05-28

"""


class mosse:
    def __init__(self, args, sequence_path):
        # get arguments..
        self.args = args
        self.sequence_path = sequence_path
        self.img_path = join(sequence_path, 'img')
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()
    

    def track_sequence(self):
        results = []
        gt_boxes = load_gt(join(self.sequence_path, 'groundtruth_rect.txt'), delimiter='\t')
        # get the image of the first frame... (read as gray scale image...)
        init_img = cv2.imread(self.frame_lists[0])
        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)
        # get the init ground truth.. [x, y, width, height]
        init_gt = gt_boxes[0]
        init_gt = np.array(init_gt).astype(np.int64)
        init_gt_w = init_gt[2]
        init_gt_h = init_gt[3]
        # start to draw the gaussian response...
        response_map = self._get_gauss_response(init_frame, init_gt)
        # start to create the training set ...
        # get the goal..
        g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        g = pad_img(g, FFT_SIZE)
        fi = init_frame[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        G = np.fft.fft2(g)
        # start to do the pre-training...
        Ai, Bi = self._pre_training(fi, G)
        # start the tracking...
        for idx in range(len(self.frame_lists)):
            current_frame = cv2.imread(self.frame_lists[idx])
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)
            if idx == 0:
                Ai = self.args.lr * Ai
                Bi = self.args.lr * Bi
                pos = init_gt.copy()
                clip_pos = np.array([pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]).astype(np.int64)
            else:
                Hi = Ai / Bi
                # print('Hi shape:', Hi.shape, Hi.dtype)
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])), padded_size=FFT_SIZE)
                Gi = Hi * np.fft.fft2(fi)
                gi = np.real(linear_mapping(np.fft.ifft2(Gi)))
                # print('gi:', gi.shape, gi.dtype, type(gi))
                # cv2.imshow('predicted gaussian', gi)
                # cv2.waitKey(0)
                # find the max pos...
                max_value = np.max(gi)
                max_pos = np.where(gi == max_value)
                # print('maxpos:', max_pos)
                dy = int(np.mean(max_pos[0]) - init_gt_h / 2)
                dx = int(np.mean(max_pos[1]) - init_gt_w / 2)
                
                # update the position...
                pos[0] = pos[0] + dx
                pos[1] = pos[1] + dy

                # trying to get the clipped position [xmin, ymin, xmax, ymax]
                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0]+pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1]+pos[3], 0, current_frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                # get the current fi..
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])), padded_size=FFT_SIZE)
                # online update...
                Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
                Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi
            
            results.append(pos.copy())
            # cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
            # cv2.imshow('demo', current_frame)
            # if cv2.waitKey(0) == ord('q'):
            #     break
        return results


    # start to do the object tracking...
    def start_tracking(self):
        # get the image of the first frame... (read as gray scale image...)
        init_img = cv2.imread(self.frame_lists[0])
        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)
        # get the init ground truth.. [x, y, width, height]
        init_gt = cv2.selectROI('demo', init_img, False, False)
        init_gt = np.array(init_gt).astype(np.int64)
        init_gt_w = init_gt[2]
        init_gt_h = init_gt[3]
        # start to draw the gaussian response...
        response_map = self._get_gauss_response(init_frame, init_gt)
        # start to create the training set ...
        # get the goal..
        g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        g = pad_img(g, FFT_SIZE)
        fi = init_frame[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        G = np.fft.fft2(g)
        # start to do the pre-training...
        Ai, Bi = self._pre_training(fi, G)
        # start the tracking...
        for idx in range(len(self.frame_lists)):
            current_frame = cv2.imread(self.frame_lists[idx])
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)
            if idx == 0:
                Ai = self.args.lr * Ai
                Bi = self.args.lr * Bi
                pos = init_gt.copy()
                clip_pos = np.array([pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]).astype(np.int64)
            else:
                Hi = Ai / Bi
                # print('Hi shape:', Hi.shape, Hi.dtype)
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])), padded_size=FFT_SIZE)
                Gi = Hi * np.fft.fft2(fi)
                gi = np.real(linear_mapping(np.fft.ifft2(Gi)))
                # print('gi:', gi.shape, gi.dtype, type(gi))
                # cv2.imshow('predicted gaussian', gi)
                # cv2.waitKey(0)
                # find the max pos...
                max_value = np.max(gi)
                max_pos = np.where(gi == max_value)
                # print('maxpos:', max_pos)
                dy = int(np.mean(max_pos[0]) - init_gt_h / 2)
                dx = int(np.mean(max_pos[1]) - init_gt_w / 2)
                
                # update the position...
                pos[0] = pos[0] + dx
                pos[1] = pos[1] + dy

                # trying to get the clipped position [xmin, ymin, xmax, ymax]
                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0]+pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1]+pos[3], 0, current_frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                # get the current fi..
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])), padded_size=FFT_SIZE)
                # online update...
                Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
                Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi
            
            # visualize the tracking process...
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
            cv2.imshow('demo', current_frame)
            if cv2.waitKey(100) == ord('q'):
                break
            # if record... save the frames..
            if self.args.record:
                frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
                if not os.path.exists(frame_path):
                    os.mkdir(frame_path)
                cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)


    # pre train the filter on the first frame...
    def _pre_training(self, init_frame, G):
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))
        # print('fi:', fi.shape, init_frame.shape, np.unique(fi-init_frame) )
        # pre-process img..
        fi = pre_process(fi, padded_size=FFT_SIZE)
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = pre_process(random_warp(init_frame), padded_size=FFT_SIZE)
            else:
                fi = pre_process(init_frame, padded_size=FFT_SIZE)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        
        return Ai, Bi

    # get the ground-truth gaussian reponse...
    def _get_gauss_response(self, img, gt):
        # get the shape of the image..
        height, width = img.shape
        # get the mesh grid...
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # get the center of the object...
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)
        return response

    # it will extract the image list 
    def _get_img_lists(self, img_path):
        frame_list = []
        for frame in os.listdir(img_path):
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(img_path, frame)) 
        return frame_list
    
    # it will get the first ground truth of the video..
    def _get_init_ground_truth(self, img_path):
        gt_path = os.path.join(img_path, 'groundtruth.txt')
        with open(gt_path, 'r') as f:
            # just read the first frame...
            line = f.readline()
            gt_pos = line.split(',')

        return [float(element) for element in gt_pos]

