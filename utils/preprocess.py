from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
import math


def rgb2gray(rgb):
    gray_image     = 0.2126 * rgb[..., 0]
    gray_image[:] += 0.0722 * rgb[..., 1]
    gray_image[:] += 0.7152 * rgb[..., 2]
    return gray_image

def rgb2y(rgb):
    y_image     = 0.299 * rgb[..., 0]
    y_image[:] += 0.587 * rgb[..., 1]
    y_image[:] += 0.114 * rgb[..., 2]
    return y_image

def scale(image, hei_image, wid_image):
    return cv2.resize(image, (wid_image, hei_image), interpolation=cv2.INTER_LINEAR)

def preprocessAtari(frame, crop1=34, crop2=34, mean=0.369, std=0.155):
    frame = frame[crop1:crop2 + 160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = rgb2gray(frame)
    frame = frame.astype(np.float32)
    frame*= (1. / 255.)
    frame = (frame-mean)/std
    return frame

def preprocessAtariRGB(frame,crop1, crop2, mean, std, running_mean=False):
    frame = frame[crop1:crop2 + 160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = frame.astype(np.float32)
    frame*= (1. / 255.)
    frame = np.transpose(frame,(2,0,1))
    frame[0] = (frame[0]-mean[0])/std[0]
    frame[1] = (frame[1]-mean[1])/std[1]
    frame[2] = (frame[2]-mean[2])/std[2]
    if running_mean:
        alpha = 0.9999
        deltas = []
        deltas.append(frame[0].mean() - mean[0])
        deltas.append(frame[1].mean() - mean[1])
        deltas.append(frame[2].mean() - mean[2])
        mean[0] = mean[0]  + deltas[0] * (1-alpha)
        mean[1] = mean[1]  + deltas[1] * (1-alpha)
        mean[2] = mean[2]  + deltas[2] * (1-alpha)
        std[0]  = math.sqrt(alpha * (std[0] * std[0] + (1-alpha) * deltas[0] * deltas[0]))
        std[1]  = math.sqrt(alpha * (std[1] * std[1] + (1-alpha) * deltas[1] * deltas[1]))
        std[2]  = math.sqrt(alpha * (std[2] * std[2] + (1-alpha) * deltas[2] * deltas[2]))
    return frame, mean, std

def preprocessAtariRGB_simple(frame,crop1, crop2, mean, std, running_mean=False):
    frame = frame[crop1:crop2 + 160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = frame.astype(np.float32)
    frame*= (1. / 255.)
    frame = np.transpose(frame,(2,0,1))
    frame = frame * 2 - 1
    return frame, mean, std

def postprocessAtariRGB(processed_frame ,  mean, std):
    frame = processed_frame.copy()
    frame[0] = frame[0] * std[0] + mean[0]
    frame[1] = frame[1] * std[1] + mean[1]
    frame[2] = frame[2] * std[2] + mean[2]
    frame = np.transpose(frame, (2,0,1))
    frame = np.transpose(frame, (2,0,1))
    frame = frame * 255
    frame = frame.astype('uint8')
    return frame

def zoomAtariRGB(processed_frame, mean, std):
    '''
    this function zooms in the atari frame
    to between -1 and 1.
    '''
    frame = processed_frame.copy()
    frame[0] = frame[0] * std[0] + mean[0]
    frame[1] = frame[1] * std[1] + mean[1]
    frame[2] = frame[2] * std[2] + mean[2]
    frame = frame * 2 -1 
    return frame 


def postprocessAtariRGB_v2(frame ,  mean, std):
    frame[0] = frame[0] * std[0] + mean[0]
    frame[1] = frame[1] * std[1] + mean[1]
    frame[2] = frame[2] * std[2] + mean[2]
    return frame
