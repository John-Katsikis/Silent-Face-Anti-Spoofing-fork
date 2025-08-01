# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午2:13
# @Author : zhuying
# @Company : Minivision
# @File : utility.py
# @Software : PyCharm

from datetime import datetime
import os


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-') ## returns current date formatted as 'YYYY-MM-DD-HH-MM'


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size ## adds 15 to both height and width and then performs floor division by 16, I don't know why 15 is added


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1]) ## gets horizontal length
    h_input = int(patch_info.split('x')[0].split('_')[-1]) ## gets vertical length
    return w_input,h_input


def parse_model_name(model_name): ## 2.7_80x80_MiniFASNetV2.pth
    info = model_name.split('_')[0:-1] ## ['2.7','80x80']
    h_input, w_input = info[-1].split('x') ## '80' and '80'
    model_type = model_name.split('.pth')[0].split('_')[-1] ## MiniFASNetV2

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0]) ## 2.7
    return int(h_input), int(w_input), model_type, scale ## (80, 80, 'MiniFASNetV2', 2.7)


def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
