#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : comb.py
# @Author: WangYe
# @Date  : 2020/3/27
# @Software: PyCharm
import os
import cv2
import numpy as np
path = 'output_map'
out_path = 'test.png'
ori_image = [6116, 3357, 3]
h_step = 23
w_step = 13
h_rest = -228
w_rest = -29
predict_list = []
files = os.listdir(path)
for file in range(len(files)):
    file_path = path+'/'+str(file)+'.png_result.png'
    im = cv2.imread(file_path)
    predict_list.append(im[:, :, :])
# predict_list.append(ori_image[-256:, -256:, :])
count_temp = 0
tmp = np.ones([ori_image[0], ori_image[1],ori_image[2]])
for h in range(h_step):
    for w in range(w_step):
        tmp[h * 256:(h + 1) * 256,w * 256:(w + 1) * 256,:] = predict_list[count_temp]
        count_temp += 1
    tmp[h * 256:(h + 1) * 256, w_rest:,:] = predict_list[count_temp][:, w_rest:,:]
    count_temp += 1
for w in range(w_step - 1):
    tmp[h_rest:, (w * 256):(w * 256 + 256),:] = predict_list[count_temp][h_rest:, :,:]
    count_temp += 1
tmp[h_rest:, w_rest:,:] = predict_list[count_temp][h_rest:, w_rest:,:]
print(tmp.shape)
cv2.imwrite(out_path,tmp)
