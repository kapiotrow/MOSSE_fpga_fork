import numpy as np
import os
from os.path import join
import time
import sys
import torch
import torchvision.transforms as transforms
from easydict import EasyDict

from fxpmath import Fxp
import cv2

from utils import get_VGG_backbone, linear_mapping, random_warp, pad_img, load_gt, window_func_2d, pre_process, get_CF_backbone
from imagenet.finn_models import get_finnlayer



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


def quantize_fi(x, signed, int_bits, frac_bits, outfile=None):

    # print(str(x.dtype), x[-1, 0, 0], 'complex' in str(x.dtype))
    width = int_bits + frac_bits
    x = Fxp(x, signed, width, frac_bits)
    
    if outfile:
        content = 'memory_initialization_radix=2;\n'
        content += 'memory_initialization_vector=\n'

        for row in x:
            for el in row:
                if 'complex' in str(x.dtype):
                    bin_repr = el.bin()[:-1]
                    bin_repr = bin_repr.replace('+', '')
                    bin_repr = bin_repr.replace('-', '')
                    bin_real = bin_repr[0:width]
                    bin_imag = bin_repr[width:]
                    content += bin_imag + bin_real + '\n'
                else:
                    content += el.bin() + '\n'
        content += ';'

        with open(outfile, 'w') as result_file:
            result_file.write(content)
        sys.exit()
    

class DeepMosse:
    def __init__(self, init_frame, init_position, config, debug=False):
        # self.backbone = get_CF_backbone(net_config_path, net_weights_path)
        args = EasyDict(config)
        args.debug = debug
        print('config:', args)
        if args.deep:
            self.backbone_device = torch.device('cuda:0')
            if args.quantized:
                print('ITS SO DEEP QUANTIZED')
                self.use_quant_features = True
                self.quant_scaling = np.load('/home/vision/danilowi/CF_tracking/MOSSE_fpga/deployment/Mul_0_param0.npy')
                self.backbone = get_finnlayer('/home/vision/danilowi/CF_tracking/MOSSE_fpga/output/finnlayer_weights/savegame_0_15000.pth.tar', strict=False)
            else:
                self.backbone = get_VGG_backbone()
            self.backbone.to(self.backbone_device)
            self.stride = 2
        else:
            # print('your regular boy')
            self.stride = 1
        self.features_width = args.ROI_SIZE//self.stride            

        #buffer features during prediction for update
        self.buffer_features_for_update = args.buffer_features
        self.buffered_features_shape = (args.num_scales,
                                        args.channels,
                                        args.ROI_SIZE//self.stride + 2*args.buffered_padding,
                                        args.ROI_SIZE//self.stride + 2*args.buffered_padding)
        self.buffered_features = np.zeros(self.buffered_features_shape, dtype=np.float32)
        self.buffered_windows = np.zeros((args.num_scales,
                                          2*args.buffered_padding*self.stride + args.ROI_SIZE,
                                          2*args.buffered_padding*self.stride + args.ROI_SIZE,
                                          3), dtype=np.uint8)
        # print('buffer features: {}, padding {}'.format(args.buffer_features, args.buffered_padding))
        # print("buffered windows dtype:", self.buffered_windows.dtype)
        # print("buffered features dtype:", self.buffered_features.dtype)

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

        self.args = args
        self.initialize(init_frame, init_position)
        self.current_frame = 1


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
        self.x_scale = (self.args.ROI_SIZE/self.stride) / (xmax - xmin)
        self.y_scale = (self.args.ROI_SIZE/self.stride) / (ymax - ymin)

        if self.buffer_features_for_update and not ignore_buffering:
            # computing additional context in image dimensions to achieve <self.args.buffered_padding> in features dimensions
            # and maintain self.x_scale, self.y_scale
            window_widen = 2 * self.args.buffered_padding * self.stride
            dw = (self.args.ROI_SIZE + window_widen) / (self.x_scale*self.stride) - (xmax - xmin)
            dh = (self.args.ROI_SIZE + window_widen) / (self.y_scale*self.stride) - (ymax - ymin)
            dw /= 2
            dh /= 2
            xmin = np.clip(xmin-dw, 0, frame.shape[1])
            xmax = np.clip(xmax+dw, 0, frame.shape[1])
            ymin = np.clip(ymin-dh, 0, frame.shape[0])
            ymax = np.clip(ymax+dh, 0, frame.shape[0])
            window = frame[int(round(ymin)) : int(round(ymax)), int(round(xmin)) : int(round(xmax)), :]
            # print('window shape:', window.shape)
            # print('window position:', ymin-dh, ymax+dh)
            window = cv2.resize(window, (self.args.ROI_SIZE + window_widen, self.args.ROI_SIZE + window_widen))
            # cropped_window = window[self.stride*self.args.buffered_padding : -self.stride*self.args.buffered_padding,
            #                         self.stride*self.args.buffered_padding : -self.stride*self.args.buffered_padding,
            #                         :]
            self.buffered_windows[scale_idx] = window
            if self.args.debug:
                cv2.imshow('{} wider search window {:.3f}'.format(debug, scale), window.astype(np.uint8))
                # if debug == 'predict' and self.current_frame <= 10:
                    # cv2.imwrite('../test_inputs/test_input_{}x{}_fpad{}_BGR_frame{:2d}.ppm'.format(self.ROI_SIZE, self.ROI_SIZE, self.args.buffered_padding, self.current_frame), cv2.cvtColor(window, cv2.COLOR_RGB2BGR))
                    # cv2.waitKey(0)
                    # sys.exit()
        else:
            window = frame[int(round(ymin)) : int(round(ymax)), int(round(xmin)) : int(round(xmax)), :]
            window = cv2.resize(window, (self.args.ROI_SIZE, self.args.ROI_SIZE))
            if self.args.debug:
                cv2.imshow('{} search window {:.3f}'.format(debug, scale), window.astype(np.uint8))
            # np.save('test_input_256x256.npy', window)

        window = self.extract_features(window)
 
        if self.buffer_features_for_update and not ignore_buffering:
            self.buffered_features[scale_idx] = window
            window = window[:,
                            self.args.buffered_padding : -self.args.buffered_padding,
                            self.args.buffered_padding : -self.args.buffered_padding]

        return window


    def extract_features(self, window):

        if self.args.deep:
            # print(window[:, :, 0])
            # print(window[:, :, 1])
            # print(window[:, :, 2])
            window = self.cnn_preprocess(window)
            window = self.backbone(window)[0].detach()
            window = window.cpu().numpy()
            window = window[:self.args.channels, :, :]
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

        g = self._get_gauss_response(self.args.ROI_SIZE//self.stride)
        # cv2.imshow('goal', (g*255).astype(np.uint8))
        G = np.fft.fft2(g)
        # G_real = np.real(G)
        # G_imag = np.imag(G)
        # quantize_fi(G_real, True, 9, 23, 'deployment/memory_inits/gauss32_64x64.coe')
        # sys.exit()
        # print('range:', )
        # print('unique:', np.unique(G_imag))
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
            Hi_real = np.real(self.Hi)*4096
            Hi_imag = np.imag(self.Hi)*4096
            # print(self.Hi)
            # print(self.Ai.shape, self.Bi.shape, self.Hi.shape)
            # print('maxes:')
            # print(np.max(np.abs(np.real(self.Hi))), np.max(np.abs(np.imag(self.Hi))))
            # quantize_fi(self.Hi[0], True, 0, 32, outfile='deployment/memory_inits/coef32_64x64_ch0.coe')
            # sys.exit()
            # print('ai:', self.Ai[:, 0, 0])
            # print('bi:', self.Bi[0, 0])
            # print('hi:', self.Hi[:, 0, 0])

        # position in [x1, y1, w, h]
        # clip_pos in [x1, y1, x2, y2]
        self.position = init_gt.copy()
        # print('frame 0:', (0, 0))
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
            hi_real = np.real(self.Hi)*4096
            hi_imag = np.imag(self.Hi)*4096
            fftfi = np.fft.fft2(fi)
            fftfi_real = np.real(fftfi)/4096
            fftfi_imag = np.imag(fftfi)/4096

            Gi = self.Hi * np.fft.fft2(fi)
            gi_real = np.real(Gi)
            gi_imag = np.imag(Gi)
            Gi = np.sum(Gi, axis=0)

            Gi_real = np.real(Gi)
            Gi_imag = np.imag(Gi)
            # print('dżi:', Gi.shape)
            gi = np.real(np.fft.ifft2(Gi))
            gi_real = gi/4096

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
                    print('frame {}:'.format(self.current_frame), features_displacement)

        self.position = best_position
        # print('position:', self.position)


    def update(self, frame):

        if self.buffer_features_for_update:
            # window = self.buffered_windows[self.best_scale_idx]
            # x_win = self.args.buffered_padding*self.stride + self.best_features_displacement[0]*self.stride
            # y_win = self.args.buffered_padding*self.stride + self.best_features_displacement[1]*self.stride
            # cropped_window = window[y_win : y_win+self.args.ROI_SIZE, 
            #                         x_win : x_win+self.args.ROI_SIZE,
            #                         :]
            # cv2.imshow('cropped search window', cropped_window.astype(np.uint8))
            # fi = self.extract_features(cropped_window)
            fi = self.buffered_features[self.best_scale_idx]
            x_position_in_window = self.args.buffered_padding + self.best_features_displacement[0]
            y_position_in_window = self.args.buffered_padding + self.best_features_displacement[1]

            if np.abs(self.best_features_displacement[0]) > self.args.buffered_padding or np.abs(self.best_features_displacement[1]) > self.args.buffered_padding:
                print('DISPLACEMENT {}, {} BIGGER THAN FEATURES PADDING, CLIPPING...'.format(*self.best_features_displacement))
                x_position_in_window = np.clip(x_position_in_window, 0, 2*self.args.buffered_padding)
                y_position_in_window = np.clip(y_position_in_window, 0, 2*self.args.buffered_padding)
            
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
            # xd = fftfi * np.conjugate(fftfi)
            # print('fftfi:', xd[0, :10, :10])
            # dBSUM_lr_channels_tdata = np.real(self.args.lr * (np.sum(fftfi * np.conjugate(fftfi) + self.args.lambd, axis=0))) / (4096*4096)
            self.Ai = self.args.lr * (self.G * np.conjugate(fftfi)) + (1 - self.args.lr) * self.Ai
            self.Bi = self.args.lr * (np.sum(fftfi * np.conjugate(fftfi) + self.args.lambd, axis=0)) + (1 - self.args.lr) * self.Bi
            # ASUM_treal = np.real(self.Ai)/4096
            # BSUM_treal = np.real(self.Bi)/(4096*4096)
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
            self.current_frame += 1

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
            # fftfi_real = np.real(fftfi)/4096
            # fftfi_imag = np.imag(fftfi)/4096
            Ai = G * np.conjugate(fftfi)
            # Ai_real = np.real(Ai)/4096
            # Ai_imag = np.imag(Ai)/4096
            Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) + self.args.lambd
            # Bi_real = np.real(Bi)/(4096*4096)
            # Bi_imag = np.imag(Bi)/(4096*4096)
            Bi = Bi.sum(axis=0)
            # Bisum_real = np.real(Bi)/(4096*4096)
            # print('bi:', Bi.shape)
            # print('ai:', Ai.shape)

        # for _ in range(self.args.num_pretrain):
        #     if self.args.rotate:
        #         fi = self.pre_process(random_warp(template, str(_)))
        #     else:
        #         fi = self.pre_process(template)

        #     if self.use_fixed_point:
        #         fftfi = Fxp(np.fft.fft2(fi), *self.fxp_precision).get_val()
        #         Ai = Fxp(Ai + G * np.conjugate(fftfi), *self.fxp_precision).get_val()
        #         Bi = Fxp(Bi + fftfi * np.conjugate(fftfi), *self.fxp_precision).get_val()
        #     else:
        #         fftfi = np.fft.fft2(fi)
        #         Ai = Ai + np.conjugate(G) * fftfi
        #         Bi = Bi + np.sum(np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) + self.args.lambd, axis=0) + self.args.lambd
        #         # Bi = Bi + np.sum(np.fft.fft2(fi), axis=0) * np.sum(np.conjugate(np.fft.fft2(fi)), axis=0)
                

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
        # quantize_fi(window, False, 0, 32, "my_hann32_{}x{}.coe".format(height, width))
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

