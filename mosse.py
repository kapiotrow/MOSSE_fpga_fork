import numpy as np
import cv2
import os
from os.path import join
import time


from utils import linear_mapping, random_warp, pad_img, load_gt, window_func_2d, pre_process


# FFT_SIZE = 200


"""
This module implements the basic correlation filter based tracking algorithm -- MOSSE

Date: 2018-05-28

"""


class mosse:
    def __init__(self, args, sequence_path, FFT_SIZE=0):
        # get arguments..
        self.args = args
        self.sequence_path = sequence_path
        self.img_path = join(sequence_path, 'img')
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()
        self.pad_type = 'center'
        self.big_search_window = True
        self.FFT_SIZE = FFT_SIZE


    #bbox: [xmin, ymin, xmax, ymax]
    def crop_search_window(self, bbox, frame, size):
        if size != 0:
            xmin, ymin, xmax, ymax = bbox
            xdiff = size - xmax + xmin
            ydiff = size - ymax + ymin
            # print('diffs:', xdiff, ydiff)
            win_xmin = xmin - xdiff // 2
            win_xmax = xmax + xdiff // 2
            if xdiff % 2 != 0:
                win_xmax += 1
            win_ymin = ymin - ydiff // 2
            win_ymax = ymax + ydiff // 2
            if ydiff % 2 != 0:
                win_ymax += 1

            # print('window:', win_xmin, win_xmax, win_ymin, win_ymax)
            # to padded frame coordinates:
            win_xmin += size
            win_xmax += size
            win_ymin += size
            win_ymax += size

            padded_frame = cv2.copyMakeBorder(frame, size, size, size, size, cv2.BORDER_CONSTANT)
            # cv2.imshow('padded frame', padded_frame.astype(np.uint8))
            window = padded_frame[win_ymin : win_ymax, win_xmin : win_xmax]
            # print('window shape:', window.shape)
        else:
            window = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        if self.args.debug:
            cv2.imshow('search window', window.astype(np.uint8))

        return window


    # start to do the object tracking...
    def start_tracking(self):
        results = []
        # get the image of the first frame... (read as gray scale image...)
        gt_boxes = load_gt(join(self.sequence_path, 'groundtruth.txt'))
        init_img = cv2.imread(self.frame_lists[0])
        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)
        # get the init ground truth.. [x, y, width, height]
        # init_gt = cv2.selectROI('demo', init_img, False, False)
        init_gt = gt_boxes[0]
        init_gt = np.array(init_gt).astype(np.int64)
        # init_gt[2] -= init_gt[2] % 2
        # init_gt[3] -= init_gt[3] % 2
        init_gt_w = init_gt[2]
        init_gt_h = init_gt[3]
        maxdim = max(init_gt_w, init_gt_h)
        if maxdim > self.FFT_SIZE and self.FFT_SIZE != 0:
            # print('Warning, FFT_SIZE changed to ', maxdim)
            self.FFT_SIZE = maxdim
        # start to draw the gaussian response...
        response_map = self._get_gauss_response(init_frame, init_gt)
        # start to create the training set ...
        # get the goal..
        g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        g, _ = pad_img(g, self.FFT_SIZE, pad_type=self.pad_type)
        # cv2.imshow('goal', g)
        # cv2.waitKey(0)
        G = np.fft.fft2(g)
        # start to do the pre-training...
        Ai, Bi = self._pre_training(init_gt, init_frame, G)
        # start the tracking...
        start_time = time.time()
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
                if self.big_search_window:
                    fi = self.crop_search_window(clip_pos, frame_gray, self.FFT_SIZE)
                    fi = self.pre_process(fi)
                else:
                    fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                    fi = self.pre_process(fi, padded_size=self.FFT_SIZE)
                Gi = Hi * np.fft.fft2(fi)
                gi = np.real(linear_mapping(np.fft.ifft2(Gi)))

                if self.args.debug:
                    cv2.imshow('response', (gi*255).astype(np.uint8))   
                # print('gi:', gi.shape, gi.dtype, type(gi))
                # cv2.imshow('predicted gaussian', gi)
                # cv2.waitKey(0)
                # find the max pos...
                max_value = np.max(gi)
                max_pos = np.where(gi == max_value)
                if not self.big_search_window and self.FFT_SIZE != 0 and self.pad_type == 'topleft':
                    dy = int(np.mean(max_pos[0]) - init_gt_h / 2) if len(max_pos[0]) != 0 else 0
                    dx = int(np.mean(max_pos[1]) - init_gt_w / 2) if len(max_pos[1]) != 0 else 0
                else:
                    dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2) if len(max_pos[0]) != 0 else 0
                    dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2) if len(max_pos[1]) != 0 else 0

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
                if self.big_search_window:
                    fi = self.crop_search_window(clip_pos, frame_gray, self.FFT_SIZE)
                    fi = self.pre_process(fi)
                else:
                    fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                    fi = self.pre_process(fi, padded_size=self.FFT_SIZE)
                # online update...
                Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
                Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi
            
            results.append(pos.copy())
            if self.args.debug:
                cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
                cv2.imshow('demo', current_frame)
                if cv2.waitKey(0) == ord('q'):
                    break

        print(len(self.frame_lists)/(time.time() - start_time), 'fps')
            

        return results


    # pre train the filter on the first frame...
    def _pre_training(self, init_gt, init_frame, G):
        padded_size = self.FFT_SIZE if not self.big_search_window else 0

        if self.big_search_window:
            bbox = [init_gt[0], init_gt[1], init_gt[0]+init_gt[2], init_gt[1]+init_gt[3]]
            template = self.crop_search_window(bbox, init_frame, self.FFT_SIZE)
        else:
            template = init_frame[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]

        fi = self.pre_process(template, padded_size=padded_size)
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = self.pre_process(random_warp(template), padded_size=padded_size)
            else:
                fi = self.pre_process(template, padded_size=padded_size)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        
        return Ai, Bi


    # pre-processing the image...
    def pre_process(self, img, padded_size=0, use_window=True):
        # get the size of the img...
        # xd, _ = pad_img(img, padded_size, pad_type=self.pad_type)
        # cv2.imshow('padded', xd.astype(np.uint8))

        height, width = img.shape
        img = np.log(img + 1)
        img = (img - np.mean(img)) / (np.std(img) + 1e-5)

        if use_window:
            # use the hanning window...
            window = window_func_2d(height, width)
            img = img * window

        img, _ = pad_img(img, padded_size, pad_type=self.pad_type)

        return img


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



class mosse_old:
    def __init__(self, args, sequence_path):
        # get arguments..
        self.args = args
        self.sequence_path = sequence_path
        self.img_path = join(sequence_path, 'img')
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()
        self.target_lost = False
    
    # start to do the object tracking...
    def start_tracking(self):
        results = []
        # get the image of the first frame... (read as gray scale image...)
        gt_boxes = load_gt(join(self.sequence_path, 'groundtruth.txt'))
        init_img = cv2.imread(self.frame_lists[0])
        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)
        # get the init ground truth.. [x, y, width, height]
        # init_gt = cv2.selectROI('demo', init_img, False, False)
        init_gt = gt_boxes[0]
        init_gt = np.array(init_gt).astype(np.int64)
        # start to draw the gaussian response...
        response_map = self._get_gauss_response(init_frame, init_gt)
        # start to create the training set ...
        # get the goal..
        g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
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
            elif self.target_lost:
                pos = [0, 0, 0, 0]
            else:
                Hi = Ai / Bi
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                Gi = Hi * np.fft.fft2(fi)
                gi = linear_mapping(np.fft.ifft2(Gi))
                # find the max pos...
                max_value = np.max(gi)
                max_pos = np.where(gi == max_value)
                dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
                dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
                
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
                # print('fi shape:', fi.shape)
                if min(fi.shape) == 0:
                    self.target_lost = True
                    print('TARGET LOST')
                else:
                    fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                    # online update...
                    Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
                    Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi
            
            results.append(pos.copy())
            # visualize the tracking process...
            # cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
            # cv2.imshow('demo', current_frame)
            # if cv2.waitKey(0) == ord('q'):
            #     break
            # if record... save the frames..
            if self.args.record:
                frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
                if not os.path.exists(frame_path):
                    os.mkdir(frame_path)
                cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)

        return results


    # pre train the filter on the first frame...
    def _pre_training(self, init_frame, G):
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))
        # pre-process img..
        fi = pre_process(fi)
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                warp = random_warp(init_frame)
                fi = pre_process(warp)
            else:
                fi = pre_process(init_frame)
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
