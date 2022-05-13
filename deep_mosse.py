import numpy as np
import os
from os.path import join
import time
import sys
import torch
import torchvision.transforms as transforms

from fxpmath import Fxp
import cv2

from utils import get_VGG_backbone, linear_mapping, random_warp, pad_img, load_gt, window_func_2d, pre_process, get_CF_backbone
from imagenet.finn_models import get_finnlayer

# FFT_SIZE = 200


"""
This module implements the basic correlation filter based tracking algorithm -- MOSSE

Date: 2018-05-28

"""

def quant(x, mul):
    q = np.round(x / mul)

    # test = np.load('output_packed_valid.npy')
    # test = test.squeeze()
    # print(q.shape, test.shape)
    # print('same:', np.unique(np.isclose(q, test.transpose(2, 0, 1))))
    # print(q[:, 0, 0])
    
    return q


def quantize_param(x, bits, range, outfile):

    scale = range / (2**bits - 1)
    q = np.round(x / scale)
    q = q.flatten().astype(np.uint32)
    result = ''.join([np.binary_repr(el, width=bits) + '\n' for el in q])[:-1]
    with open(outfile, 'w') as result_file:
        result_file.write(result)


def quantize_fi(x, signed, int_bits, frac_bits, outfile):

    x = Fxp(x, signed, int_bits + frac_bits, frac_bits)
    test = x.get_val()
    content = 'memory_initialization_radix=2;\n'
    content += 'memory_initialization_vector=\n'
    sys.exit()

    for row in x:
        for el in row:
            content += el.bin() + '\n'
    content += ';'

    with open(outfile, 'w') as result_file:
        result_file.write(content)
    


class DeepMosse:
    def __init__(self, init_frame, init_position, args, FFT_SIZE=224, buffer_features=True):
        # self.backbone = get_CF_backbone(net_config_path, net_weights_path)
        if args.deep:
            self.backbone_device = torch.device('cuda:0')
            self.channels = 64
            if args.quantized:
                print('ITS SO DEEP QUANTIZED')
                self.use_quant_features = True
                self.quant_scaling = np.load('/home/magister/CF_tracking/MOSSE_fpga/Mul_0_param0.npy')
                self.backbone = get_finnlayer('/home/magister/CF_tracking/MOSSE_fpga/output/finnlayer_weights/savegame_0_15000.pth.tar', self.channels, strict=False)
            else:
                self.backbone = get_VGG_backbone()
            self.backbone.to(self.backbone_device)
            self.stride = 2
        else:
            # print('your regular boy')
            self.channels = 1
            self.stride = 1
        self.features_width = FFT_SIZE//self.stride            

        #buffer features during prediction for update
        self.buffer_features_for_update = buffer_features
        self.buffered_padding = 64  #in features dimensions
        self.buffered_features_shape = (args.num_scales,
                                        self.channels,
                                        FFT_SIZE//self.stride + 2*self.buffered_padding,
                                        FFT_SIZE//self.stride + 2*self.buffered_padding)
        self.buffered_features = np.zeros(self.buffered_features_shape, dtype=np.float32)
        self.buffered_windows = np.zeros((args.num_scales,
                                          2*self.buffered_padding*self.stride + FFT_SIZE,
                                          2*self.buffered_padding*self.stride + FFT_SIZE,
                                          3), dtype=np.uint8)
        print('buffer features: {}, padding {}'.format(buffer_features, self.buffered_padding))
        # print("buffered windows dtype:", self.buffered_windows.dtype)
        # print("buffered features dtype:", self.buffered_features.dtype)

        self.args = args
        self.FFT_SIZE = FFT_SIZE
        scale_exponents = [i - np.floor(args.num_scales / 2) for i in range(args.num_scales)]
        self.scale_multipliers = [pow(args.scale_factor, ex) for ex in scale_exponents]
        self.target_lost = False
        if args.border_type == 'constant':
            self.border_type = cv2.BORDER_CONSTANT
        elif args.border_type == 'reflect':
            self.border_type = cv2.BORDER_REFLECT
        elif args.border_type == 'replicate':
            self.border_type = cv2.BORDER_REPLICATE

        self.use_fixed_point = False
        self.fractional_precision = 8
        self.fxp_precision = [True, 31+self.fractional_precision, self.fractional_precision]

        self.initialize(init_frame, init_position)


    #bbox: [xmin, ymin, w, h]
    def crop_search_window(self, bbox, frame, scale=1, debug='test', scale_idx=0, ignore_buffering=False):
        
        xmin, ymin, width, height = bbox
        # print('bbox:', bbox)
        xmax = xmin + width
        ymax = ymin + height
        if self.args.search_region_scale != 1:
            x_offset = (width * scale * self.args.search_region_scale - width) / 2
            y_offset = (height * scale * self.args.search_region_scale - height) / 2
            xmin = xmin - x_offset
            xmax = xmax + x_offset
            ymin = ymin - y_offset
            ymax = ymax + y_offset

        if not self.args.clip_search_region:
            x_pad = int(width * self.args.search_region_scale)
            y_pad = int(height * self.args.search_region_scale)
            frame = cv2.copyMakeBorder(frame, y_pad, y_pad, x_pad, x_pad, self.border_type)
            xmin += x_pad
            xmax += x_pad
            ymin += y_pad
            ymax += y_pad
        xmin = np.clip(xmin, 0, frame.shape[1])
        xmax = np.clip(xmax, 0, frame.shape[1])
        ymin = np.clip(ymin, 0, frame.shape[0])
        ymax = np.clip(ymax, 0, frame.shape[0])

        # scaling from image dimensions to features dimensions - for calculating displacement later
        self.x_scale = (self.FFT_SIZE/self.stride) / (xmax - xmin)
        self.y_scale = (self.FFT_SIZE/self.stride) / (ymax - ymin)

        if self.buffer_features_for_update and not ignore_buffering:
            # computing additional context in image dimensions to achieve <self.buffered_padding> in features dimensions
            # and maintain self.x_scale, self.y_scale
            window_widen = 2*self.buffered_padding*self.stride
            dw = (self.FFT_SIZE + window_widen) / (self.x_scale*self.stride) - (xmax - xmin)
            dh = (self.FFT_SIZE + window_widen) / (self.y_scale*self.stride) - (ymax - ymin)
            dw /= 2
            dh /= 2
            xmin = np.clip(xmin-dw, 0, frame.shape[1])
            xmax = np.clip(xmax+dw, 0, frame.shape[1])
            ymin = np.clip(ymin-dh, 0, frame.shape[0])
            ymax = np.clip(ymax+dh, 0, frame.shape[0])
            window = frame[int(round(ymin)) : int(round(ymax)), int(round(xmin)) : int(round(xmax)), :]
            # print('window shape:', window.shape)
            # print('window position:', ymin-dh, ymax+dh)
            window = cv2.resize(window, (self.FFT_SIZE + window_widen, self.FFT_SIZE + window_widen))
            # cropped_window = window[self.stride*self.buffered_padding : -self.stride*self.buffered_padding,
            #                         self.stride*self.buffered_padding : -self.stride*self.buffered_padding,
            #                         :]
            self.buffered_windows[scale_idx] = window
            if self.args.debug:
                cv2.imshow('{} wider search window {:.3f}'.format(debug, scale), window.astype(np.uint8))
        else:
            window = frame[int(round(ymin)) : int(round(ymax)), int(round(xmin)) : int(round(xmax)), :]
            window = cv2.resize(window, (self.FFT_SIZE, self.FFT_SIZE))
            if self.args.debug:
                cv2.imshow('{} search window {:.3f}'.format(debug, scale), window.astype(np.uint8))
            # np.save('test_input_256x256.npy', window)
            # cv2.imwrite('test_input_{}x{}_BGR.ppm'.format(self.FFT_SIZE, self.FFT_SIZE), cv2.cvtColor(window, cv2.COLOR_RGB2BGR))
            # sys.exit()

        window = self.extract_features(window)
 
        if self.buffer_features_for_update and not ignore_buffering:
            self.buffered_features[scale_idx] = window
            window = window[:,
                            self.buffered_padding : -self.buffered_padding,
                            self.buffered_padding : -self.buffered_padding]

        return window


    def extract_features(self, window):

        if self.args.deep:
            # print(window[:, :, 0])
            # print(window[:, :, 1])
            # print(window[:, :, 2])
            window = self.cnn_preprocess(window)
            window = self.backbone(window)[0].detach()
            window = window.cpu().numpy()
            if self.args.quantized and self.use_quant_features:
                window = quant(window, self.quant_scaling)
                # print('out:', window[0].shape, window[0])
        else:
            window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            window = np.expand_dims(window, axis=2)
            # print('the shape:', window.shape)
            window = window.transpose(2, 0, 1)

        return window


    def initialize(self, init_frame, init_position):
       
        self.frame_shape = init_frame.shape
        init_gt = init_position
        init_gt = np.array(init_gt).astype(np.int64)

        g = self._get_gauss_response(self.FFT_SIZE//self.stride)
        # cv2.imshow('goal', (g*255).astype(np.uint8))
        G = np.fft.fft2(g)
        if self.use_fixed_point:
            G = Fxp(G, *self.fxp_precision)
            G = np.array(G)
        # start to do the pre-training...
        Ai, Bi = self._pre_training(init_gt, init_frame, G)

        self.Ai = Ai
        self.Bi = Bi
        self.G = G

        if self.use_fixed_point:
            # Ai = Fxp(Ai, *self.fxp_precision)
            # Bi = Fxp(Bi, *self.fxp_precision)
            self.Hi = Fxp(self.Ai / self.Bi, *self.fxp_precision)
        else:
            self.Hi = self.Ai / self.Bi
            # print(self.Ai.shape, self.Bi.shape, self.Hi.shape)
            # print('ai:', self.Ai[:, 0, 0])
            # print('bi:', self.Bi[0, 0])
            # print('hi:', self.Hi[:, 0, 0])

        # position in [x1, y1, w, h]
        # clip_pos in [x1, y1, x2, y2]
        self.position = init_gt.copy()
        # self.clip_pos = np.array([self.position[0], self.position[1], self.position[0]+self.position[2], self.position[1]+self.position[3]]).astype(np.int64)


    def predict(self, frame, position, scale=1, scale_idx=None):

        fi = self.crop_search_window(position, frame, scale, debug='predict', scale_idx=scale_idx)
        # print('predict shape:', fi.shape)
        fi = self.pre_process(fi)
        
        # print(self.Hi.dtype, self.Hi.shape)

        if self.use_fixed_point:
            # print('Hi_fixed', Fxp(Hi).info())
            Gi = self.Hi * Fxp(np.fft.fft2(fi), *self.fxp_precision).get_val()
            # Gi.info()
            gi = np.real(np.fft.ifft2(Gi))
        else:
            Gi = np.conjugate(self.Hi) * np.fft.fft2(fi)
            Gi = np.sum(Gi, axis=0)
            # print('dżi:', Gi.shape)
            gi = np.real(np.fft.ifft2(Gi))

        if self.args.debug:
            cv2.imshow('response', gi)   

        return gi


    def predict_multiscale(self, frame):

        best_response = 0
        for scale_idx, scale in enumerate(self.scale_multipliers):
            # print('scale:', scale)
            response = self.predict(frame, self.position, scale, scale_idx=scale_idx)
            new_position, max_response, features_displacement = self.update_position(response, scale)
            if max_response > best_response:
                best_response = max_response
                best_position = new_position
                if self.buffer_features_for_update:
                    self.best_scale_idx = scale_idx
                    self.best_features_displacement = features_displacement

        self.position = best_position
        # print('position:', self.position)


    def update(self, frame):

        if self.buffer_features_for_update:
            # window = self.buffered_windows[self.best_scale_idx]
            # x_win = self.buffered_padding*self.stride + self.best_features_displacement[0]*self.stride
            # y_win = self.buffered_padding*self.stride + self.best_features_displacement[1]*self.stride
            # cropped_window = window[y_win : y_win+self.FFT_SIZE, 
            #                         x_win : x_win+self.FFT_SIZE,
            #                         :]
            # cv2.imshow('cropped search window', cropped_window.astype(np.uint8))
            # fi = self.extract_features(cropped_window)
            fi = self.buffered_features[self.best_scale_idx]
            x_position_in_window = self.buffered_padding + self.best_features_displacement[0]
            y_position_in_window = self.buffered_padding + self.best_features_displacement[1]

            if np.abs(self.best_features_displacement[0]) > self.buffered_padding or np.abs(self.best_features_displacement[1]) > self.buffered_padding:
                print('DISPLACEMENT {}, {} BIGGER THAN FEATURES PADDING, CLIPPING...'.format(*self.best_features_displacement))
                x_position_in_window = np.clip(x_position_in_window, 0, 2*self.buffered_padding)
                y_position_in_window = np.clip(y_position_in_window, 0, 2*self.buffered_padding)
            
            fi = fi[:,
                    y_position_in_window : y_position_in_window+self.features_width,
                    x_position_in_window : x_position_in_window+self.features_width]
        else:
            fi = self.crop_search_window(self.position, frame, debug='update', ignore_buffering=True)
            # print('update features dtype:', fi.dtype)
        fi = self.pre_process(fi)

        if self.use_fixed_point:
            fftfi = Fxp(np.fft.fft2(fi), *self.fxp_precision)
            self.Ai = self.args.lr * (self.G * np.conjugate(fftfi)) + (1 - self.args.lr) * self.Ai
            self.Bi = self.args.lr * fftfi * np.conjugate(fftfi) + (1 - self.args.lr) * self.Bi
            # Ai.info()
            # Bi.info()
            # Ai = Fxp(Ai.get_val(), *self.fxp_precision)
            # Bi = Fxp(Bi.get_val(), *self.fxp_precision)
            self.Hi = Fxp(self.Ai.get_val() / self.Bi.get_val(), *self.fxp_precision)
        else:
            fftfi = np.fft.fft2(fi)
            xd = fftfi * np.conjugate(fftfi)
            # print('fftfi:', xd[0, :10, :10])
            self.Ai = self.args.lr * (np.conjugate(self.G) * fftfi) + (1 - self.args.lr) * self.Ai
            self.Bi = self.args.lr * (np.sum(fftfi * np.conjugate(fftfi) + self.args.lambd, axis=0)) + (1 - self.args.lr) * self.Bi
            self.Hi = self.Ai / self.Bi



    def update_position(self, spatial_response, scale=1):

        gi = spatial_response
        max_value = np.max(gi)
        max_pos = np.where(gi == max_value)

        dy = np.mean(max_pos[0]) - gi.shape[0] / 2
        dx = np.mean(max_pos[1]) - gi.shape[1] / 2
        features_displacement = (int(dx), int(dy))
        dx /= self.x_scale
        dy /= self.y_scale

        new_width = self.position[2]*scale
        new_height = self.position[3]*scale
        dw = new_width - self.position[2]
        dh = new_height - self.position[3]

        new_xmin = self.position[0] + dx - dw/2
        new_ymin = self.position[1] + dy - dh/2
        new_position = [new_xmin, new_ymin, new_width, new_height]

        return new_position, max_value, features_displacement


    def check_position(self):
        # # trying to get the clipped position [xmin, ymin, xmax, ymax]
        clip_xmin = np.clip(self.position[0], 0, self.frame_shape[1])
        clip_ymin = np.clip(self.position[1], 0, self.frame_shape[0])
        clip_xmax = np.clip(self.position[0] + self.position[2], 0, self.frame_shape[1])
        clip_ymax = np.clip(self.position[1] + self.position[3], 0, self.frame_shape[0])
        if clip_xmax-clip_xmin == 0 or clip_ymax-clip_ymin == 0:
            self.target_lost = True
        # self.clip_pos = self.clip_pos.astype(np.int64)


    def track(self, image):
        
        if not self.target_lost:
            # response = self.predict(image, self.position)
            self.predict_multiscale(image)
            # self.update_position(response)
            self.check_position()

            if not self.target_lost:
                self.update(image)

        return [int(el) for el in self.position]
        

    # pre train the filter on the first frame...
    def _pre_training(self, init_gt, init_frame, G):

        template = self.crop_search_window(init_gt, init_frame)
        # quant(template, 'Mul_0_param0.npy')
        # print('template:', template)
        fi = self.pre_process(template)

        if self.use_fixed_point:
            fftfi = Fxp(np.fft.fft2(fi), *self.fxp_precision).get_val()
            Ai = Fxp(G * np.conjugate(fftfi), *self.fxp_precision).get_val()
            Bi = Fxp(fftfi * np.conjugate(fftfi)).get_val()
        else:
            fftfi = np.fft.fft2(fi)
            Ai = np.conjugate(G) * fftfi
            Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) + self.args.lambd
            Bi = Bi.sum(axis=0)
            # print('bi:', Bi.shape)
            # print('ai:', Ai.shape)

        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = self.pre_process(random_warp(template, str(_)))
            else:
                fi = self.pre_process(template)

            if self.use_fixed_point:
                fftfi = Fxp(np.fft.fft2(fi), *self.fxp_precision).get_val()
                Ai = Fxp(Ai + G * np.conjugate(fftfi), *self.fxp_precision).get_val()
                Bi = Fxp(Bi + fftfi * np.conjugate(fftfi), *self.fxp_precision).get_val()
            else:
                fftfi = np.fft.fft2(fi)
                Ai = Ai + np.conjugate(G) * fftfi
                Bi = Bi + np.sum(np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) + self.args.lambd, axis=0) + self.args.lambd
                # Bi = Bi + np.sum(np.fft.fft2(fi), axis=0) * np.sum(np.conjugate(np.fft.fft2(fi)), axis=0)
                

        return Ai, Bi


    # pre-processing the image...
    def pre_process(self, img):
        # get the size of the img...
        # xd, _ = pad_img(img, padded_size, pad_type=self.pad_type)
        # cv2.imshow('padded', xd.astype(np.uint8))
        channels, height, width = img.shape
        # print(type(img), img.shape)
        # img = np.log(img + 1)
        # print('img:', img)
        # img = (img - np.mean(img)) / (np.std(img) + 1e-5)

        window = window_func_2d(height, width)
        # quantize_param(window, bits=32, range=1, outfile="my_hann32_{}x{}.coe".format(height, width))
        # quantize_fi(window, False, 0, 16, "my_hann16_{}x{}.coe".format(height, width))
        # sys.exit()
        # print('window:', window.shape)
        if self.use_fixed_point:
            img = Fxp(img, *self.fxp_precision) * Fxp(window, *self.fxp_precision)
            img = np.array(img)
        else:
            img = img * window

        # img, _ = pad_img(img, padded_size, pad_type=self.pad_type)

        return img


    def cnn_preprocess(self, data):

        # result = data.copy()
        # result.resize(1, data.shape[0], data.shape[1], data.shape[2])
        # result = result.transpose(0, 3, 1, 2)
        # result = torch.from_numpy(result)
        # result = result.float() / 255.0

        # print('input:', data.shape)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        result = transform(data)
        result = result.unsqueeze(dim=0)
        result = result.to(self.backbone_device)

        # print('result:', result.device)

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


    def check_clip_pos(self):
        width = self.clip_pos[2] - self.clip_pos[0]
        height = self.clip_pos[3] - self.clip_pos[1]

        return width > 0 and height > 0

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
