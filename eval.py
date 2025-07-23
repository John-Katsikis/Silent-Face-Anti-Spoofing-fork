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

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"

def test(model_dir, device_id, path):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    type = path.split("/")[-1]
    print(type)
    
    endresult = open("result.txt", "a")
    realface = 0
    fakeface = 0
    size = 0
    count = 0
    
    for image_name in os.listdir(path):
        if image_name.endswith('.png'):
            count+=1
            full_path = os.path.join(path, image_name)
            image = cv2.imread(full_path)
            if image is None:
                print(f"Image {full_path} failed to load")
                continue

            image_bbox = model_test.get_bbox(image)
            prediction = np.zeros((1, 3))
            test_speed = 0
            size+=1
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
            value = prediction[0][label] / 2  # optional: normalize or rescale score
            print(f"VALUE IS {value}")

            if label == 1:
                endresult.write("Image '{}' is Real Face. Score: {:.2f}.\n".format(image_name, value))
                realface+=1
                print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
                result_text = "RealFace Score: {:.2f}".format(value)
                color = (255, 0, 0)
            else:
                endresult.write("Image '{}' is Fake Face. Score: {:.2f}.\n".format(image_name, value))
                fakeface+=1
                print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
                result_text = "FakeFace Score: {:.2f}".format(value)
                color = (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(
                image,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                color, 2)
            cv2.putText(
                image,
                result_text,
                (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color)

            # Save the result image
            format_ = os.path.splitext(image_name)[-1]
            result_image_name = image_name.replace(format_, "_result" + format_)
            cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)

    endresult.close()
    
    if "Live" in path:
        correctness = (realface/size)*100
        falsePos = size - realface
        print("Live Correctness rate is : {:.2f}%".format(correctness))
        print(f"Precision is: {realface/(realface+falsePos)}")
    
    if "Spoof" in path:
        correctness = (fakeface/size)*100
        print("Spoof Correctness rate is : {:.2f}%".format(correctness))
    
    print(f"{realface} faces are real, {fakeface} are fake.")

    print(f"{realface + fakeface} images overall. ")
    print("All TXT results written.")
    
    
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
    parser.add_argument(
        "--eval_path",
        type=str,
        default="EVALSET/Live",
        help="filepath for evaluation dataset")
    args = parser.parse_args()
    test(args.model_dir, args.device_id, args.eval_path)
