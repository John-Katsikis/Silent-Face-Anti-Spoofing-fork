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

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"

labels = {
    "Unknown": 0,
    "Live": 1,
    "Spoof": 2
}

directory_labels = {
    "Live": 1,
    "Spoof": 2
}



def test(model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    directories = ["EVALSET/Live", "EVALSET/Spoof"]

    photonum=0

    truthFull = []
    predFull = []

    endresult = open("result.txt", "a")

    for path in directories:
        type = path.split("/")[-1]
        print(f"Running {type}")
        
        realface = 0
        fakeface = 0
        size = 0
        count = 0
        
        truthList = []
        predList = []

        for image_name in os.listdir(path):
            truth_label = labels[type]

            if image_name.endswith('.png'):
                
                photonum+=1
                truthList.append(truth_label)
                truthFull.append(truth_label)
                
                count += 1
                full_path = os.path.join(path, image_name)
                image = cv2.imread(full_path)
                if image is None:
                    print(f"Image {full_path} failed to load")
                    continue

                image_bbox = model_test.get_bbox(image)
                prediction = np.zeros((1, 3))
                test_speed = 0
                size += 1
                for model_name in os.listdir(model_dir):
                    h_input, w_input, model_type, scale = parse_model_name(model_name)
                    param = {
                        "org_img": image,
                        "bbox": image_bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }
                    if scale is None:
                        param["crop"] = False
                    img = image_cropper.crop(**param)
                    start = time.time()
                    prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                    test_speed += time.time() - start

                label = np.argmax(prediction)
                value = prediction[0][label] / 2    
                
                
                if(label==0):
                    print(f"Image {image_name} is Unknown")
                    print(value)

                predList.append(label)
                predFull.append(label)

                  # optional: normalize or rescale score
               
                overallMatrix = confusion_matrix(truthList, predList, labels=[0, 1, 2])
                precision = precision_score(truthList, predList, average='micro')
                recall = recall_score(truthList, predList, average='micro')
                f1 = f1_score(truthList, predList, average='micro')

        print(overallMatrix)
        print(f"Precision: {precision*100}%")
        print(f"Recall: {recall*100}%")
        print(f"F1 Score: {f1*100}%")

    overallFullMatrix = confusion_matrix(truthFull, predFull, labels=[0, 1, 2])
    overallFullPrecision = precision_score(truthFull, predFull, average='micro')
    overallFullRecall = recall_score(truthFull, predFull, average='micro')
    overallFullF1 = f1_score(truthFull, predFull, average='micro')
    print("Overall Full Matrix:")
    print(overallFullMatrix)
    print(f"Overall Full Precision: {overallFullPrecision*100}%")
    print(f"Overall Full Recall: {overallFullRecall*100}%")
    print(f"Overall Full F1 Score: {overallFullF1*100}%")
    print(f"Total images processed: {photonum}")
  



    endresult.close()


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
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg",
        help="image used to test")
    args = parser.parse_args()
    test(args.model_dir, args.device_id)
