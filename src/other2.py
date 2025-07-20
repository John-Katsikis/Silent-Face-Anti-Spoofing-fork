# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoof_predict.py
# @Software : PyCharm

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}


class Detection:
    def __init__(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel) ## Loads retinaface model
        self.detector_confidence = 0.4

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1] ## Gets height and width of input image
        aspect_ratio = width / height ## Calculates aspect ratio of image
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img, ## Image resizing if it's too large, to avoid unecessary processing
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR) ## Bilinear interpolation for downsizing

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123)) ## Converts image to "blob", to be used in a deep neural net with OpenCV
        self.detector.setInput(blob, 'data') ## Sets input of the detector to be the blob from the last line
        out = self.detector.forward('detection_out').squeeze() ## Performs forward inference
       
        detections = out[out[:, 2] > self.detector_confidence] ## Filters out detections where the confidence is lower than the confidence threshold defined in line 33
      
        bboxes = [] ## initializes empty list to return
      
        if len(detections) == 0: ## if no faces are detected , grab small slice of image and feed that into the anti spoofing model for testing purposes
            bbox= [0, 0, 80, 80]
            bboxes.append(bbox)

        for detection in detections:
            left, top, right, bottom = detection[3]*width, detection[4]*height, \
                                       detection[5]*width, detection[6]*height
            bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
            print(left, top, right, bottom)
            bboxes.append(bbox)         ## Converts bounding box coordinates to pixel coordinates
        
        return bboxes


class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id) ## Checks if CUDA is available, if not then CPU is chosen
                                   if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path) ## extracts filename from the filepath
        h_input, w_input, model_type, _ = parse_model_name(model_name) ## extracts height,width and name of model
        self.kernel_size = get_kernel(h_input, w_input,) ## calls get_kernel method found in utility.py ot determine the size of the kernel to be used
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device) ## instantiates a model, passing any needed parameters and passing it off to a specified device

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device) ## loads the saved model weights, and on to the correct device
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0: ## checks if first key contains the substring 'module.'
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items(): ## iterates over key-value pairs from original state_dict
                name_key = key[7:] ## removes the first 7 characters from the name, likely the 'module.' string
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict) ## loads the new state dict to the model
        else:
            self.model.load_state_dict(state_dict) ## if no 'moduke.' prefix was found, load the original state dict 
        return None

    def predict(self, img, model_path):
        test_transform = trans.Compose([
            trans.ToTensor(), ## Converts image to PyTorch tensor
        ])
        img = test_transform(img) ## Applies transform pipeline to image
        img = img.unsqueeze(0).to(self.device) ## Adds extra dimension at the front to symbolize batch size as PyTorch requires [C,H,W] to [1,C,H,W]
        self._load_model(model_path) ## Loads model weights from pth files
        self.model.eval() ## Sets model to evaluation mode
        with torch.no_grad():
            result = self.model.forward(img) ## Forward inference
            result = F.softmax(result).cpu().numpy() ## Converts raw scores (logits) into probabilities
        return result

