# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import torch

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

# Custom AntiSpoofPredict class with MPS support
class MPSAntiSpoofPredict(AntiSpoofPredict):
    def __init__(self, device_id):
        # Call parent constructor but override device selection
        super().__init__(device_id)
        
        # Override device selection to prefer MPS (Apple Neural Engine)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.load_device = torch.device("cpu")  # Load on CPU, run on MPS
            print("Using Apple Neural Engine (MPS) for inference, CPU for loading")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(device_id))
            self.load_device = self.device
            print(f"Using CUDA GPU {device_id}")
        else:
            self.device = torch.device("cpu")
            self.load_device = self.device
            print("Using CPU")
    
    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        from src.utility import get_kernel
        from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
        
        MODEL_MAPPING = {
            'MiniFASNetV1': MiniFASNetV1,
            'MiniFASNetV2': MiniFASNetV2,
            'MiniFASNetV1SE': MiniFASNetV1SE,
            'MiniFASNetV2SE': MiniFASNetV2SE
        }
        
        self.kernel_size = get_kernel(h_input, w_input)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size)

        # Load model weight on CPU first, then move to target device
        state_dict = torch.load(model_path, map_location=self.load_device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        
        # Move model to target device after loading
        self.model = self.model.to(self.device)
        return None


SAMPLE_IMAGE_PATH = "./images/sample/"


def test_webcam(model_dir, device_id):
    model_test = MPSAntiSpoofPredict(device_id)
    image_cropper = CropImage()

    camera = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    
    while True:
        flag, frame = camera.read()
        
        # Check if frame was read successfully
        if not flag:
            print("Error: Could not read frame from camera")
            break

        # Skip or remove any check_image calls for now

        image_bbox = model_test.get_bbox(frame)
        if image_bbox is None:
            # No face detected; just show the frame and continue
            cv2.imshow('Anti-Spoof Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        prediction = np.zeros((1, 3))
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": frame,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))

        label = np.argmax(prediction)
        text = 'Real Face' if label == 1 else 'Fake Face'

        # Draw bbox and label
        x, y, w, h = image_bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        if label == 1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

       

        cv2.imshow('livetest', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    args = parser.parse_args()
    test_webcam(args.model_dir, args.device_id)
