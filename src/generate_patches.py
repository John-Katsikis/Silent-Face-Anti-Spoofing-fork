# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm
"""
Create patch from original input image by using bbox coordinate
"""

import cv2
import numpy as np


class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3] ## unzips box into top left coordinates and width/height

        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale)) ## ensures that scaling doesn't become larger than the image itself

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y ## recalculates new size after rescaling and calculates center coordinates of the new box

        left_top_x = center_x-new_width/2
        left_top_y = center_y-new_height/2
        right_bottom_x = center_x+new_width/2
        right_bottom_y = center_y+new_height/2 ## recalculates corners based on the center coordinates and new size

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1

        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1
            
            ## above if statements ensure that the boxes aren't drawn outside the image, deals with edge cases more specifically

        return int(left_top_x), int(left_top_y),\
               int(right_bottom_x), int(right_bottom_y) ## returns bounding box as integers

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h)) ## Runs if crop variable is set to false, resizes image to specified dimensions: either upscales or downscales
        else:
            src_h, src_w, _ = np.shape(org_img) ## extracts original image resolution , the _ is to ignore the number of channels
            left_top_x, left_top_y, \
                right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale) ## this method essentially scales up the bbox to capture edge details of the face like the chin or the forehead but safely so the new box doesn't cause problems

            img = org_img[left_top_y: right_bottom_y+1,
                          left_top_x: right_bottom_x+1] ## crops original image with new coordinates in mind
            dst_img = cv2.resize(img, (out_w, out_h)) ## resizes cropped image
        return dst_img
