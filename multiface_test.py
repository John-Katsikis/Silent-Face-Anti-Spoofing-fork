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

from src.other2 import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


def test_multiface_webcam(model_dir, device_id):
    model_test = AntiSpoofPredict(device_id) ##Starts up other2.py file that uses both Retinaface and anti spoofing models
    image_cropper = CropImage()

    camera = cv2.VideoCapture(0) ## Starts up camera
    
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    
    # Create window and center it on screen
    cv2.namedWindow('multi-face test', cv2.WINDOW_AUTOSIZE)
    # Center the window (approximate values for common screen sizes)
    cv2.moveWindow('multi-face test', 300, 100)
    
    while True:
        ret, frame = camera.read()
        
        # Check if frame was read successfully
        if not ret:
            print("Error: Could not read frame from camera")
            break

        # Get all face bounding boxes (returns a list in this version)
        image_bboxes = model_test.get_bbox(frame)
        
        if image_bboxes is None:
            # No faces detected; show the frame and continue
            print('No faces detected')
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('multi-face test', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # For loop to process each detected face
        for i, image_bbox in enumerate(image_bboxes):
            prediction = np.zeros((1, 3))
            
         
            # For loop to run predictions on each model in the file directory, file directory can either be found in line 117 or if non-standard filepath then it can be found in the argument that user passed
            for model_name in os.listdir(model_dir):
                if not model_name.endswith('.pth'):
                    continue
                    
                h_input, w_input, model_type, scale = parse_model_name(model_name) ## Dictionary packing, code that performs packing can be found in utility.py line 27
                param = {
                    "org_img": frame, ## image
                    "bbox": image_bbox, ## bbox to crop out
                    "scale": scale, ## scaling factor
                    "out_w": w_input, ## Horizontal resolution
                    "out_h": h_input, ## Vertical resolution
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param) ## dictionary unpacking to image cropper, takes bbox and crops that out, to be fed later to Anti-Spoofing model
                
                prediction += model_test.predict(img, os.path.join(model_dir, model_name)) ## Forward inference on Anti-Spoofing model, works by finding the model specified by model_name which is declard in the second for loop in line 52
               

            label = np.argmax(prediction) ## index 1 means real face, index 2 means fake face, index 0 unknown (potentially unsure)
            
            confidence = np.max(prediction) ## Returns largest value in array
            
            print(f"Prediction : {prediction}  --> Chosen index: {label}  confidence: {confidence}")
           
            text = 'Real Face' if label == 1 else 'Fake Face'

            # Draw bbox and label for this face
            x, y, w, h = image_bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # Color code: Green for real, Red for fake
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            
            # Draw rectangle and circle around face
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            ##cv2.circle(frame, (int(w/2+x1), int(h/2+y1)), int(w/2), color, 2)
            
            label_text = f"{text} #{i+1}"
            cv2.putText(frame, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            

        # Resize frame to 720p for display (makes window smaller)
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('multi-face test', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    desc = "Multi-face anti-spoofing detection"
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
    test_multiface_webcam(args.model_dir, args.device_id)
