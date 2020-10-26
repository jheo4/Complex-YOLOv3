from __future__ import print_function

import numpy as np
import utils.config as cnf
import cv2
import os

class ObjectContext():
    def __init__(self):
        self.xyz = None
        self.ry = None
        self.cls_pred = None


class DetectorOptions():
    def __init__(self,
            class_path='data/classes.names',
            model_def='config/complex_tiny_yolov3.cfg',
            weights_path='checkpoints/tiny-yolov3_ckpt_epoch-220.pth',
            folder='training',
            split='valid',
            img_size=cnf.BEV_WIDTH,
            conf_thres=0.5,
            nms_thres=0.5):

        self.class_path=class_path
        self.conf_thres=conf_thres
        self.folder=folder
        self.img_size=img_size
        self.nms_thres=nms_thres
        self.split=split
        self.model_def=model_def
        self.weights_path=weights_path

