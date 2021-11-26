import numpy as np
import os
from os.path import join
import time
import sys
import torch

from fxpmath import Fxp
import cv2

from utils import get_VGG_backbone, linear_mapping, random_warp, pad_img, load_gt, window_func_2d, pre_process, get_CF_backbone


# FFT_SIZE = 200


"""
This module implements the basic correlation filter based tracking algorithm -- MOSSE

Date: 2018-05-28

"""


class DeepMosse:
    def __init__(self, args, sequence_path, net_config_path, net_weights_path, FFT_SIZE=0):
        # get arguments..
        print('its so deep')
        # self.backbone = get_CF_backbone(net_config_path, net_weights_path)
        self.backbone = get_VGG_backbone()
        self.stride = 2
        # sys.exit()

        self.args = args
        self.sequence_path = sequence_path
        self.img_path = join(sequence_path, 'img')
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()
        self.pad_type = 'center'
        self.big_search_window = True
        self.FFT_SIZE = FFT_SIZE
        self.use_fixed_point = False
        self.fractional_precision = 8
        self.fxp_precision = [True, 31+self.fractional_precision, self.fractional_precision]


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
            # padded_frame = torch.nn.functional.pad(frame, (size, size, size, size))
            # cv2.imshow('padded frame', padded_frame.astype(np.uint8))
            # cv2.waitKey(0)
            window = padded_frame[win_ymin : win_ymax, win_xmin : win_xmax, :]
            cnn_window = self.cnn_preprocess(window)
            # print('inwindow:', window.shape)
            cnn_window = self.backbone(cnn_window)[0].detach()
            # print('window shape:', cnn_window.shape)
            # print('padded frame:', padded_frame.shape)
            # sys.exit()
        else:
            window = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        if self.args.debug:
            cv2.imshow('search window', window.astype(np.uint8))

        # cv2.waitKey(0)

        return cnn_window.numpy()


    def initialize(self):

        # get the image of the first frame... (read as gray scale image...)
        gt_boxes = load_gt(join(self.sequence_path, 'groundtruth.txt'))
        init_frame = cv2.imread(self.frame_lists[0])
        # print('in img:', init_frame.shape)
        # init_frame = self.cnn_preprocess(init_frame)
        # img_features = self.backbone(init_frame)[0].detach()
        # print('out features:', img_features.shape)
        # sys.exit()

        # get the init ground truth.. [x, y, width, height]
        # init_gt = cv2.selectROI('demo', init_img, False, False)
        # transform init bboxes from image space to cnn features space
        init_gt = gt_boxes[0]
        # self.x_scale = img_features.shape[2] / init_img.shape[1]
        # self.y_scale = img_features.shape[1] / init_img.shape[0]
        # init_gt[0] *= self.x_scale
        # init_gt[1] *= self.y_scale
        # init_gt[2] *= self.x_scale
        # init_gt[3] *= self.y_scale
        init_gt = np.array(init_gt).astype(np.int64)
        # init_gt[2] -= init_gt[2] % 2
        # init_gt[3] -= init_gt[3] % 2
        init_gt_w = init_gt[2]
        init_gt_h = init_gt[3]
        maxdim = max(init_gt_w, init_gt_h)
        if maxdim > self.FFT_SIZE and self.FFT_SIZE != 0:
            # print('Warning, FFT_SIZE changed to ', maxdim)
            self.FFT_SIZE = maxdim
        # init_frame = img_features

        # start to draw the gaussian response...
        g = self._get_gauss_response(self.FFT_SIZE//self.stride)
        # get the goal..
        # g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        # g, _ = pad_img(g, self.FFT_SIZE/self.stride, pad_type=self.pad_type)
        cv2.imshow('goal', (g*255).astype(np.uint8))
        G = np.fft.fft2(g)
        if self.use_fixed_point:
            G = Fxp(G, *self.fxp_precision)
            G = np.array(G)
        # start to do the pre-training...
        Ai, Bi = self._pre_training(init_gt, init_frame, G)
        self.init_gt = init_gt

        return Ai, Bi, G


    def predict(self, frame, clip_pos, Hi):

        if self.big_search_window:
            fi = self.crop_search_window(clip_pos, frame, self.FFT_SIZE)
            fi = self.pre_process(fi)
        else:
            fi = frame[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
            fi = self.pre_process(fi, padded_size=self.FFT_SIZE)
        
        if self.use_fixed_point:
            # print('Hi_fixed', Fxp(Hi).info())
            Gi = Hi * Fxp(np.fft.fft2(fi), *self.fxp_precision).get_val()
            # Gi.info()
            gi = np.real(np.fft.ifft2(Gi))
        else:
            Gi = Hi * np.fft.fft2(fi)
            Gi = np.sum(Gi, axis=0)
            # print('dżi:', Gi.shape)
            gi = np.real(np.fft.ifft2(Gi))

        if self.args.debug:
            cv2.imshow('response', (gi*255).astype(np.uint8))   

        return gi


    def update(self, frame, clip_pos, Ai, Bi, G):
        # get the current fi..
        if self.big_search_window:
            fi = self.crop_search_window(clip_pos, frame, self.FFT_SIZE)
            fi = self.pre_process(fi)
        else:
            fi = frame[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
            fi = self.pre_process(fi, padded_size=self.FFT_SIZE)
        # online update...
        if self.use_fixed_point:
            fftfi = Fxp(np.fft.fft2(fi), *self.fxp_precision)
            Ai = self.args.lr * (G * np.conjugate(fftfi)) + (1 - self.args.lr) * Ai
            Bi = self.args.lr * fftfi * np.conjugate(fftfi) + (1 - self.args.lr) * Bi
            # Ai.info()
            # Bi.info()
            # Ai = Fxp(Ai.get_val(), *self.fxp_precision)
            # Bi = Fxp(Bi.get_val(), *self.fxp_precision)
            Hi = Fxp(Ai.get_val() / Bi.get_val(), *self.fxp_precision)
        else:
            Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
            Bi = self.args.lr * np.sum(np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)), axis=0) + (1 - self.args.lr) * Bi
            Hi = Ai / Bi

        return Ai, Bi, Hi


    # start to do the object tracking...
    def start_tracking(self):
        results = []
        Ai, Bi, G = self.initialize()

        # print('Ai:', Ai.shape)
        # print('Bi:', Bi.shape)
        # self.use_fixed_point = False
        # Ai_fp, Bi_fp, G_fp = self.initialize()
        # print('Ai_fp:', Ai_fp)
        # Ai_diff = np.abs(Ai - Ai_fp)
        # print('mean Ai abs diff:', np.mean(Ai_diff))
        # print('Exit...')
        # sys.exit()
        
        # start the tracking...
        start_time = time.time()
        for idx in range(len(self.frame_lists)):
            current_frame = cv2.imread(self.frame_lists[idx])
            # frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            # current_frame_preproc = self.cnn_preprocess(current_frame)
            # img_features = self.backbone(current_frame_preproc)[0].detach()
            # frame_gray = frame_gray.astype(np.float32)
            if idx == 0:
                Ai = self.args.lr * Ai
                Bi = self.args.lr * Bi

                # print('Ai:', Ai.shape, type(Ai))
                # print('Bi:', Bi.shape, type(Bi))

                if self.use_fixed_point:
                    # Ai = Fxp(Ai, *self.fxp_precision)
                    # Bi = Fxp(Bi, *self.fxp_precision)
                    Hi = Fxp(Ai / Bi, *self.fxp_precision)
                else:
                    Hi = Ai / Bi

                pos = self.init_gt.copy()
                clip_pos = np.array([pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]).astype(np.int64)
            else:
                gi = self.predict(current_frame, clip_pos, Hi)
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
                pos[0] = pos[0] + dx*self.stride
                pos[1] = pos[1] + dy*self.stride

                # trying to get the clipped position [xmin, ymin, xmax, ymax]
                # print('curframe:', current_frame.shape)
                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0]+pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1]+pos[3], 0, current_frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                Ai, Bi, Hi = self.update(current_frame, clip_pos, Ai, Bi, G)
                

            result_pos = pos.copy()
            results.append(result_pos)
            if self.args.debug:
                cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
                cv2.imshow('demo', current_frame)
                if cv2.waitKey(0) == ord('q'):
                    break

        # print(len(self.frame_lists)/(time.time() - start_time), 'fps')
            

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
        if self.use_fixed_point:
            fftfi = Fxp(np.fft.fft2(fi), *self.fxp_precision).get_val()
            Ai = Fxp(G * np.conjugate(fftfi), *self.fxp_precision).get_val()
            Bi = Fxp(fftfi * np.conjugate(fftfi)).get_val()
        else:
            Ai = G * np.conjugate(np.fft.fft2(fi))
            Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
            Bi = Bi.sum(axis=0)
            # print('ai:', Ai.shape)
            # print('bi:', Bi.shape)

        for _ in range(self.args.num_pretrain):
            # print('xd:', _, end='\r')
            if self.args.rotate:
                fi = self.pre_process(random_warp(template), padded_size=padded_size)
            else:
                fi = self.pre_process(template, padded_size=padded_size)

            if self.use_fixed_point:
                fftfi = Fxp(np.fft.fft2(fi), *self.fxp_precision).get_val()
                Ai = Fxp(Ai + G * np.conjugate(fftfi), *self.fxp_precision).get_val()
                Bi = Fxp(Bi + fftfi * np.conjugate(fftfi), *self.fxp_precision).get_val()
            else:
                Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
                Bi = Bi + np.sum(np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)), axis=0)
                # Bi = Bi + np.sum(np.fft.fft2(fi), axis=0) * np.sum(np.conjugate(np.fft.fft2(fi)), axis=0)
                

        return Ai, Bi


    # pre-processing the image...
    def pre_process(self, img, padded_size=0):
        # get the size of the img...
        # xd, _ = pad_img(img, padded_size, pad_type=self.pad_type)
        # cv2.imshow('padded', xd.astype(np.uint8))

        channels, height, width = img.shape
        # print(type(img), img.shape)
        # img = np.log(img + 1)
        # print('img:', img)
        # img = (img - np.mean(img)) / (np.std(img) + 1e-5)

        window = window_func_2d(height, width)
        # print('window:', window.shape)
        if self.use_fixed_point:
            img = Fxp(img, *self.fxp_precision) * Fxp(window, *self.fxp_precision)
            img = np.array(img)
        else:
            img = img * window

        img, _ = pad_img(img, padded_size, pad_type=self.pad_type)

        # print('img shape:', img.shape)

        return img


    def cnn_preprocess(self, data):

        result = data.copy()
        result.resize(1, data.shape[0], data.shape[1], data.shape[2])
        result = result.transpose(0, 3, 1, 2)
        result = torch.from_numpy(result)
        result = result.float() / 255.0

        return result


    # get the ground-truth gaussian reponse...
    def _get_gauss_response(self, size):
       
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))
        # get the center of the object...
        center_x = size // 2
        center_y = size // 2
        
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)

        # if self.use_fixed_point:
        #     response = Fxp(response, *self.fxp_precision)
        #     response = np.array(response)

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



# class mosse_old:
#     def __init__(self, args, sequence_path):
#         # get arguments..
#         self.args = args
#         self.sequence_path = sequence_path
#         self.img_path = join(sequence_path, 'img')
#         # get the img lists...
#         self.frame_lists = self._get_img_lists(self.img_path)
#         self.frame_lists.sort()
#         self.target_lost = False
    
#     # start to do the object tracking...
#     def start_tracking(self):
#         results = []
#         # get the image of the first frame... (read as gray scale image...)
#         gt_boxes = load_gt(join(self.sequence_path, 'groundtruth.txt'))
#         init_img = cv2.imread(self.frame_lists[0])
#         init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
#         init_frame = init_frame.astype(np.float32)
#         # get the init ground truth.. [x, y, width, height]
#         # init_gt = cv2.selectROI('demo', init_img, False, False)
#         init_gt = gt_boxes[0]
#         init_gt = np.array(init_gt).astype(np.int64)
#         # start to draw the gaussian response...
#         response_map = self._get_gauss_response(init_frame, init_gt)
#         # start to create the training set ...
#         # get the goal..
#         g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
#         fi = init_frame[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
#         G = np.fft.fft2(g)
#         # start to do the pre-training...
#         Ai, Bi = self._pre_training(fi, G)
#         # start the tracking...
#         for idx in range(len(self.frame_lists)):
#             current_frame = cv2.imread(self.frame_lists[idx])
#             frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#             frame_gray = frame_gray.astype(np.float32)
#             if idx == 0:
#                 Ai = self.args.lr * Ai
#                 Bi = self.args.lr * Bi
#                 pos = init_gt.copy()
#                 clip_pos = np.array([pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]).astype(np.int64)
#             elif self.target_lost:
#                 pos = [0, 0, 0, 0]
#             else:
#                 Hi = Ai / Bi
#                 fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
#                 fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
#                 Gi = Hi * np.fft.fft2(fi)
#                 gi = linear_mapping(np.fft.ifft2(Gi))
#                 # find the max pos...
#                 max_value = np.max(gi)
#                 max_pos = np.where(gi == max_value)
#                 dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
#                 dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
                
#                 # update the position...
#                 pos[0] = pos[0] + dx
#                 pos[1] = pos[1] + dy

#                 # trying to get the clipped position [xmin, ymin, xmax, ymax]
#                 clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
#                 clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
#                 clip_pos[2] = np.clip(pos[0]+pos[2], 0, current_frame.shape[1])
#                 clip_pos[3] = np.clip(pos[1]+pos[3], 0, current_frame.shape[0])
#                 clip_pos = clip_pos.astype(np.int64)

#                 # get the current fi..
#                 fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
#                 # print('fi shape:', fi.shape)
#                 if min(fi.shape) == 0:
#                     self.target_lost = True
#                     print('TARGET LOST')
#                 else:
#                     fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
#                     # online update...
#                     Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
#                     Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi
            
#             results.append(pos.copy())
#             # visualize the tracking process...
#             # cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
#             # cv2.imshow('demo', current_frame)
#             # if cv2.waitKey(0) == ord('q'):
#             #     break
#             # if record... save the frames..
#             if self.args.record:
#                 frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
#                 if not os.path.exists(frame_path):
#                     os.mkdir(frame_path)
#                 cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)

#         return results


#     # pre train the filter on the first frame...
#     def _pre_training(self, init_frame, G):
#         height, width = G.shape
#         fi = cv2.resize(init_frame, (width, height))
#         # pre-process img..
#         fi = pre_process(fi)
#         Ai = G * np.conjugate(np.fft.fft2(fi))
#         Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
#         for _ in range(self.args.num_pretrain):
#             if self.args.rotate:
#                 warp = random_warp(init_frame)
#                 fi = pre_process(warp)
#             else:
#                 fi = pre_process(init_frame)
#             Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
#             Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        
#         return Ai, Bi

#     # get the ground-truth gaussian reponse...
#     def _get_gauss_response(self, img, gt):
#         # get the shape of the image..
#         height, width = img.shape
#         # get the mesh grid...
#         xx, yy = np.meshgrid(np.arange(width), np.arange(height))
#         # get the center of the object...
#         center_x = gt[0] + 0.5 * gt[2]
#         center_y = gt[1] + 0.5 * gt[3]
#         # cal the distance...
#         dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)
#         # get the response map...
#         response = np.exp(-dist)
#         # normalize...
#         response = linear_mapping(response)
#         return response

#     # it will extract the image list 
#     def _get_img_lists(self, img_path):
#         frame_list = []
#         for frame in os.listdir(img_path):
#             if os.path.splitext(frame)[1] == '.jpg':
#                 frame_list.append(os.path.join(img_path, frame)) 
#         return frame_list
    
#     # it will get the first ground truth of the video..
#     def _get_init_ground_truth(self, img_path):
#         gt_path = os.path.join(img_path, 'groundtruth.txt')
#         with open(gt_path, 'r') as f:
#             # just read the first frame...
#             line = f.readline()
#             gt_pos = line.split(',')

#         return [float(element) for element in gt_pos]
