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
    
    return q


def quantize_param(x, bits, range, outfile):

    scale = range / (2**bits - 1)
    q = np.round(x / scale)
    q = q.flatten().astype(np.uint32)
    result = ''.join([np.binary_repr(el, width=bits) + '\n' for el in q])[:-1]
    with open(outfile, 'w') as result_file:
        result_file.write(result)


def quantize_fi(x, signed, int_bits, frac_bits, outfile=None):

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
        args = EasyDict(config)
        args.debug = debug
        if args.deep:
            self.backbone_device = torch.device('cuda:0')
            if args.quantized:
                self.use_quant_features = False
                self.quant_scaling = np.load('/home/vision/danilowi/CF_tracking/MOSSE_fpga/deployment/Mul_0_param0.npy') if self.use_quant_features else None
                self.backbone = get_finnlayer(args.quant_weights,
                                              channels=args.channels,
                                              strict=False)
            else:
                self.backbone = get_VGG_backbone()
            self.backbone.to(self.backbone_device)
            self.stride = 2
        else:
            self.stride = 1
        self.features_width = args.ROI_SIZE//self.stride            

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

        scale_exponents = [i - np.floor(args.num_scales / 2) for i in range(args.num_scales)]
        self.scale_multipliers = [pow(args.scale_factor, ex) for ex in scale_exponents]
        self.target_not_in_bbox = False
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

        # -- DSST init
        self.base_target_sz = np.array([init_position[3], init_position[2]])
        self.nScales = self.args.nScales #number of scales (DSST)
        self.ss = np.arange(1, self.nScales+1) - np.ceil(self.nScales/2)

        self.ys = np.exp(-0.5 * np.power(self.ss, 2) / np.power(self.args.scale_sigma_factor, 2)) \
                  * 1/np.sqrt(2*np.pi * np.power(self.args.scale_sigma_factor, 2)) # desired output - gaussian-shaped peak
        self.fftys = np.fft.fft(np.reshape(self.ys, (1, self.nScales)), axis=0)

        self.currentScaleFactor = 1 # set initial scale factor to 1
        self.ss = np.arange(1, self.nScales+1)
        self.scaleFactors = np.power(self.args.scale_step, (np.ceil(self.nScales/2) - self.ss))
        self.min_scale_factor = 0.7
        self.max_scale_factor = np.power(self.args.scale_step, 
                                         (np.floor(np.log(np.min(np.divide(np.array([len(init_frame[0]), len(init_frame[1])]), self.base_target_sz)))
                                                    / np.log(self.args.scale_step))))
        if np.mod(self.nScales, 2) == 0:
            self.scale_window = np.hanning(self.nScales + 1)
            self.scale_window = self.scale_window[1:]
        else:
            self.scale_window = np.hanning(self.nScales)
        self.scale_model_factor = 1
        self.scale_model_sz = np.floor(self.base_target_sz * self.scale_model_factor).astype(int)
        self.initialize_scale(init_frame, init_position)

        # -- handling target loss
        self.min_psr = 8.7
        self.current_psr = self.min_psr + 1
        self.target_lost = False
        self.target_not_in_bbox = False
        self.target_not_in_bbox_cnt = 0


    def crop_search_window(self, bbox, frame, scale=1, debug='test', scale_idx=0, ignore_buffering=False):
        """
        Get convolutional features of the ROI.

        Args:
            bbox: [xmin, ymin, width, height] of the MOSSE translation filter
            frame: current frame
            scale: current scale factor
            debug: 
            scale_idx: 
            ignore_buffering: 
        
        Returns:
            window: convolutional features of the ROI
        """
        
        xmin, ymin, width, height = bbox
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
            window = cv2.resize(window, (self.args.ROI_SIZE + window_widen, self.args.ROI_SIZE + window_widen))
            self.buffered_windows[scale_idx] = window
            if self.args.debug:
                cv2.imshow('{} wider search window {:.3f}'.format(debug, scale), window.astype(np.uint8))
        else:
            window = frame[int(round(ymin)) : int(round(ymax)), int(round(xmin)) : int(round(xmax)), :]
            window = cv2.resize(window, (self.args.ROI_SIZE, self.args.ROI_SIZE))
            if self.args.debug:
                cv2.imshow('{} search window {:.3f}'.format(debug, scale), window.astype(np.uint8))

        window = self.extract_features(window)
 
        if self.buffer_features_for_update and not ignore_buffering:
            self.buffered_features[scale_idx] = window
            window = window[:,
                            self.args.buffered_padding : -self.args.buffered_padding,
                            self.args.buffered_padding : -self.args.buffered_padding]

        return window
    

    def crop_scale_search_window(self, pos, frame, base_target_sz, scaleFactors, scale_window, scale_model_sz):
        """
        Extract target sample.

        Extracts patches from frame, maps each to a feature vector (column) and
        concatenates them into a matrix.

        Args:
            pos: current estimated target position
            frame: current frame
            base_target_sz: size of the tracked object from the init frame
            scaleFactors: vector of scale factors
            scale_model_sz:

        Returns:
            out: matrix whose columns represent the target in different scales
        """

        for s in range(self.nScales): # iterate through all considered scale factors
            patch_sz = np.ceil(base_target_sz*scaleFactors[s]).astype(int)

            xs = np.floor(pos[0]+(pos[2]/2)) + np.arange(patch_sz[1]) - np.floor(patch_sz[1]/2)
            ys = np.floor(pos[1]+(pos[3]/2)) + np.arange(patch_sz[0]) - np.floor(patch_sz[0]/2)

            xs = [0 if i<0 else i for i in xs]
            ys = [0 if i<0 else i for i in ys]
            xs = [frame.shape[1]-1 if i>=frame.shape[1] else i for i in xs]
            ys = [frame.shape[0]-1 if i>=frame.shape[0] else i for i in ys]

            xs = np.array(xs)
            ys = np.array(ys)

            xs = xs.astype(int)
            ys = ys.astype(int)

            im_patch = frame[ys[0]:ys[-1], xs[0]:xs[-1], :]
            im_patch_resized = cv2.resize(im_patch, (scale_model_sz[1], scale_model_sz[0]), interpolation=cv2.INTER_LINEAR)
            # print(im_patch_resized.shape)
            # cv2.imshow("resized", im_patch_resized)
            # temp = self.extract_features(im_patch_resized)
            # print(temp.shape)
            temp = cv2.cvtColor(im_patch_resized, cv2.COLOR_RGB2GRAY) # use image in grayscale
            # temp = im_patch_resized[:, int(im_patch_resized.shape[1]/2), :]

            if s == 0:
                out = np.zeros((temp.size, self.nScales))

            out[:, s] = temp.flatten('F') # flatten the extracted patch features into a column vector, pack into matrix

        return out


    def extract_features(self, window):
        """
        Get convolutional features of the ROI.

        Args:
            window: ROI
        
        Returns:
            window: feature map of ROI.
        """
        
        if self.args.deep:
            window = self.cnn_preprocess(window)
            window = self.backbone(window)[0].detach()
            window = window.cpu().numpy()
            window = window[:self.args.channels, :, :]
            if self.args.quantized and self.use_quant_features:
                window = quant(window, self.quant_scaling)
        else:
            window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            window = np.expand_dims(window, axis=2)
            window = window.transpose(2, 0, 1)

        return window


    def initialize(self, init_frame, init_position) -> None:
        """
        Initialize MOSSE filter.

        Get the desire, 2D gaussian-shaped output, pre-train the MOSSE filter.

        Args:
            init_frame: first frame of the stream
            init_position: initial target position provided externally

        Returns:
            None
        """

        self.frame_shape = init_frame.shape
        init_gt = init_position
        init_gt = np.array(init_gt).astype(np.int64)

        g = self._get_gauss_response(self.args.ROI_SIZE//self.stride)
        G = np.fft.fft2(g)
        if self.use_fixed_point:
            G = Fxp(G, *self.fxp_precision)
            G = np.array(G)
        Ai, Bi = self._pre_training(init_gt, init_frame, G)

        self.Ai = Ai
        self.Bi = Bi
        self.G = G

        if self.use_fixed_point:
            self.Hi = Fxp(self.Ai / self.Bi, *self.fxp_precision)
        else:
            self.Hi = self.Ai / self.Bi
            Hi_real = np.real(self.Hi)*4096
            Hi_imag = np.imag(self.Hi)*4096
        self.position = init_gt.copy()

    
    def initialize_scale(self, init_frame, init_position) ->None:
        """
        Initialize the correlation filter used for scale estimation.

        Get the training sample xs and train the DSST correlation filter.

        Args:
            init_frame: first frame of the stream
            init_position: tracked object's position

        Returns:
            None
        """

        xs = self.crop_scale_search_window(init_position, init_frame, self.base_target_sz, self.scaleFactors, self.scale_window, self.scale_model_sz)
        fftxs = np.fft.fft(xs, axis=0)
        self.sf_num = np.multiply(fftxs, np.conjugate(self.fftys))
        self.sf_denum = np.multiply(fftxs, np.conjugate(fftxs))
        self.sf_denum = np.sum(self.sf_denum, axis=0)
        self.Yi = np.divide(self.sf_num, self.sf_denum + self.args.lambd) # DSST correlation filter
        self.target_sz = np.ceil(self.base_target_sz)


    def predict(self, frame, position, scale=1, scale_idx=None):
        """
        Predict the current target location.

        Calculates the correlation between the MOSSE filter and the current frame.

        Args:
            frame: current frame
            position: previous target position
            scale: current scale factor
            scale_idx:

        Returns:
            gi: real part of IFFT of MOSSE filter response
        """

        self.target_not_in_bbox = False # reset
        fi = self.crop_search_window(position, frame, scale, debug='predict', scale_idx=scale_idx)
        fi = self.pre_process(fi)

        if self.use_fixed_point:
            Gi = self.Hi * Fxp(np.fft.fft2(fi), *self.fxp_precision).get_val()
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
            gi = np.real(np.fft.ifft2(Gi))
            gi_real = gi/4096

        if self.args.debug:
            cv2.imshow('response', gi)

        return gi
        
    
    def predict_scale(self, frame, pos):
        """
        Predict the tracked object's scale using DSST.

        Args:
            frame: current frame
            pos: last estimated position of the tracked object

        Returns:
            scale_response: real part of IFFT of DSST filter response
        """

        xs = self.crop_scale_search_window(pos, frame, self.target_sz, self.scaleFactors, self.scale_window, self.scale_model_sz)
        fftxs = np.fft.fft(xs, axis=0)
        scale_response = np.multiply(np.conjugate(self.Yi), fftxs)
        scale_response = np.sum(scale_response, axis=0) # sum the columns
        scale_response = np.real(np.fft.ifft(np.reshape(scale_response, (1, self.nScales)), axis=0))

        return scale_response
        


    def predict_multiscale(self, frame, DSST=True) -> None:
        """
        Predict target translation and scale. 

        Combines MOSSE correlation filter on convolutional features with either 
        DSST or multiscale approach.

        Args:
            frame: current frame
            DSST: True if DSST is to be used

        Returns:
            None
        """

        if DSST: # use DSST correlation filter
            response = self.predict(frame, self.position, self.currentScaleFactor) # first find the target location
            self.position, max_response, self.best_features_displacement = self.update_position(response, self.currentScaleFactor)
            scale_response = self.predict_scale(frame, self.position) 
            self.update_scale(frame, self.position, scale_response) # then update the scale (translation usually changes faster than scale)
        
        else: # compute MOSSE CF for multiple scales and use the best response
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
            print('position:', self.position)


    def update(self, frame) -> None:
        """
        Update the numerator and denominator of the MOSSE filter; update the filter itself.

        Args:
            frme: current frame

        Returns:
            None
        """
        
        if self.buffer_features_for_update:
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
            fi = self.crop_search_window(self.position, frame, scale=self.currentScaleFactor, debug='update', ignore_buffering=True)
        fi = self.pre_process(fi)

        if self.use_fixed_point:
            fftfi = Fxp(np.fft.fft2(fi), *self.fxp_precision)
            self.Ai = self.args.lr * (self.G * np.conjugate(fftfi)) + (1 - self.args.lr) * self.Ai
            self.Bi = self.args.lr * fftfi * np.conjugate(fftfi) + (1 - self.args.lr) * self.Bi
            self.Hi = Fxp(self.Ai.get_val() / self.Bi.get_val(), *self.fxp_precision)
        else:
            fftfi = np.fft.fft2(fi)
            self.Ai = self.args.lr * (self.G * np.conjugate(fftfi)) + (1 - self.args.lr) * self.Ai
            self.Bi = self.args.lr * (np.sum(fftfi * np.conjugate(fftfi) + self.args.lambd, axis=0)) + (1 - self.args.lr) * self.Bi
            self.Hi = self.Ai / self.Bi


    def update_position(self, spatial_response, scale=1):
        """
        Update the current estimated target position.

        Checks the current PSR and updates the estimated target position if the PSR
        is greater than minimum.

        Args:
            spatial_response: response of the MOSSE filter
            scale: current scale factor
        
        Returns:
            new_position, max_vale, feature_displacement: new estimated target position,
            maximum correlation value, feature displacement
        """

        if self.target_not_in_bbox_cnt > 80:
            self.target_lost = True
            return [self.position, None, (0, 0)]
        gi = spatial_response
        max_value = np.max(gi)
        max_pos = np.where(gi == max_value)

        # check if target was located in the frame
        prev_psr_below_min = self.current_psr < self.min_psr
        self.current_psr = (max_value - np.mean(gi)) / (np.std(gi) + self.args.lambd)
        # print(self.current_psr)
        if self.current_psr < self.min_psr:
            self.target_not_in_bbox = True
            if prev_psr_below_min:
                self.target_not_in_bbox_cnt += 1
            # print("Target is not inside current bbox!")
            return [self.position, None, (0, 0)]

        self.target_not_in_bbox_cnt = 0 # reset counter
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
    
    def update_scale(self, frame, pos, scale_response):
        """
        Update the DSST correlation filter and current scale factor.

        Args:
            frame: current frame
            pos: updated target position as bbox
            scale_response: result of correlation of DSST filter and current frame

        Returns:
            None
        """

        if not self.target_not_in_bbox:
            xs = self.crop_scale_search_window(pos, frame, self.target_sz, self.scaleFactors, self.scale_window, self.scale_model_sz)
            fftxs = np.fft.fft(xs, axis=0)

            self.currentScaleFactor = self.scaleFactors[np.argmax(scale_response)] # current target scale is obtained by finding the max correlation
            self.best_scale_idx = np.argmax(scale_response)
            
            if self.currentScaleFactor > self.max_scale_factor: self.currentScaleFactor = self.max_scale_factor
            elif self.currentScaleFactor < self.min_scale_factor: self.currentScaleFactor = self.min_scale_factor

            new_sf_num = np.multiply(np.conjugate(self.fftys), fftxs)
            new_sf_den = np.sum(np.multiply(np.conjugate(fftxs), fftxs), axis=0)

            self.sf_num = (1 - self.args.lr_scale) * self.sf_num + self.args.lr_scale * new_sf_num
            self.sf_denum = (1 - self.args.lr_scale) * self.sf_denum + self.args.lr_scale * new_sf_den

            self.Yi = np.divide(self.sf_num, self.sf_denum + self.args.lambd) # update filter

            self.target_sz = np.ceil(self.target_sz * self.currentScaleFactor)



    def check_position(self):
        """
        
        """

        clip_xmin = np.clip(self.position[0], 0, self.frame_shape[1])
        clip_ymin = np.clip(self.position[1], 0, self.frame_shape[0])
        clip_xmax = np.clip(self.position[0] + self.position[2], 0, self.frame_shape[1])
        clip_ymax = np.clip(self.position[1] + self.position[3], 0, self.frame_shape[0])
        if clip_xmax-clip_xmin == 0 or clip_ymax-clip_ymin == 0:
            self.target_not_in_bbox = True


    def track(self, image, DSST=True):
        """
        Track object.

        Args:
            image: current frame
            DSST: true if DSST is used

        Returns:
            [int(el) for el in self.position]: predicted target location
        """
        
        if not self.target_lost:
            self.predict_multiscale(image, DSST)
            self.check_position()

            if not self.target_not_in_bbox:
                self.update(image)
            self.current_frame += 1

        return [int(el) for el in self.position]
        


    def _pre_training(self, init_gt, init_frame, G):
        """
        Pre-train the MOSSE filter.

        Args:
            init_gt: initial target bounding box
            init_frame: first frame of the stream
            G: desired output

        Returns:
            Ai, Bi: numerator and denominator of the MOSSE filter
        """

        template = self.crop_search_window(init_gt, init_frame)
        fi = self.pre_process(template)

        if self.use_fixed_point:
            fftfi = Fxp(np.fft.fft2(fi), *self.fxp_precision).get_val()
            Ai = Fxp(G * np.conjugate(fftfi), *self.fxp_precision).get_val()
            Bi = Fxp(fftfi * np.conjugate(fftfi)).get_val()
        else:
            fftfi = np.fft.fft2(fi)
            Ai = G * np.conjugate(fftfi)
            Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) + self.args.lambd
            Bi = Bi.sum(axis=0)

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
        #         Ai = (1 - self.args.lr) * Ai + (self.args.lr) * (G * np.conjugate(fftfi))
        #         new_Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) + self.args.lambd
        #         new_Bi = new_Bi.sum(axis=0)
        #         Bi = (1 - self.args.lr) * (self.args.lr) * (Bi + new_Bi + self.args.lambd)
        #         # Bi = Bi + np.sum(np.fft.fft2(fi), axis=0) * np.sum(np.conjugate(np.fft.fft2(fi)), axis=0)
                

        return Ai, Bi


    def pre_process(self, img):
        """
        Pre-process the image by applying 2D Hann window to it.

        Args:
            img: frame to be pre-processed

        Returns:
            img: pre-processed frame
        """

        channels, height, width = img.shape

        window = window_func_2d(height, width)
        if self.use_fixed_point:
            img = Fxp(img, *self.fxp_precision) * Fxp(window, *self.fxp_precision)
            img = np.array(img)
        else:
            img = img * window

        return img


    def cnn_preprocess(self, data):
        """
        
        """

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        result = transform(data)
        result = result.unsqueeze(dim=0)
        result = result.to(self.backbone_device)

        return result


    def _get_gauss_response(self, size):
        """
        Calculate the 2D gaussian-shaped response for MOSSE filter.

        Args:
            size: size of the gauss-shaped response

        Returns:
            response: 2D gaussian-shaped response
        """
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))
        center_x = size // 2
        center_y = size // 2
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)
        response = np.exp(-dist)
        response = linear_mapping(response)

        return response


    def _get_img_lists(self, img_path):
        frame_list = []
        for frame in os.listdir(img_path):
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(img_path, frame))
                
        return frame_list
    

    def _get_init_ground_truth(self, img_path):
        gt_path = os.path.join(img_path, 'groundtruth.txt')
        with open(gt_path, 'r') as f:
            line = f.readline()
            gt_pos = line.split(',')

        return [float(element) for element in gt_pos]


    def check_clip_pos(self):
        width = self.clip_pos[2] - self.clip_pos[0]
        height = self.clip_pos[3] - self.clip_pos[1]

        return width > 0 and height > 0

