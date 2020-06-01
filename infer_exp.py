#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Infer for ICNet model."""
from __future__ import print_function
import cityscape
import argparse
import functools
import sys
import os
import cv2
import paddle.fluid as fluid
import paddle
from icnet import icnet
from utils import add_arguments, print_arguments, get_feeder_data, check_gpu
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu
os.system("sudo rm -rf /home/wangye/wangye/icnet_paddle/cut_map/* & sudo rm -rf /home/wangye/wangye/icnet_paddle/comb_map/*")
import numpy as np
IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model_path',        str,   None,         "Model path.")
add_arg('images_list',       str,   './dataset/infer.list',         "List file with images to be infered.")
add_arg('images_path',       str,   "input.png",         "The images path.")
add_arg('out_path',          str,   "./output_map",         "Output path.")
add_arg('use_gpu',           bool,  True,       "Whether use GPU to test.")
# yapf: enable

data_shape = [3, 256, 256]
num_classes = 5

label_colours = [
    [255, 255, 255],
    [0, 255, 0],
    [0, 0, 0]
    # 0 = road, 1 = sidewalk, 2 = building
    ,
    [131, 139, 139],
    [19, 69, 139],
    [153, 153, 153]
    # 3 = wall, 4 = fence, 5 = pole
    ,
    [250, 170, 29],
    [219, 219, 0],
    [106, 142, 35]
    # 6 = traffic light, 7 = traffic sign, 8 = vegetation
    ,
    [152, 250, 152],
    [69, 129, 180],
    [219, 19, 60]
    # 9 = terrain, 10 = sky, 11 = person
    ,
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 69]
    # 12 = rider, 13 = car, 14 = truck
    ,
    [0, 60, 100],
    [0, 79, 100],
    [0, 0, 230]
    # 15 = bus, 16 = train, 17 = motocycle
    ,
    [119, 10, 32]
]

# 18 = bicycle


def color(input):
    """
    Convert infered result to color image.
    """
    result = []
    s = []
    for i in input.flatten():
        if i not in s:
            s.append(i)
       # print(i)
        result.append(
            [label_colours[i][2], label_colours[i][1], label_colours[i][0]])
    result = np.array(result).reshape([input.shape[0], input.shape[1], 3])
    # print(s)
    return result


def infer(args,cut_path,image_list,comb_path):
    data_shape = cityscape.test_data_shape()
    num_classes = cityscape.num_classes()
    # define network
    images = fluid.layers.data(name='image', shape=data_shape, dtype='float32')
    _, _, sub124_out = icnet(images, num_classes,
                             np.array(data_shape[1:]).astype("float32"))
    predict = fluid.layers.resize_bilinear(
        sub124_out, out_shape=data_shape[1:3])
    predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
    predict = fluid.layers.reshape(predict, shape=[-1, num_classes])
    _, predict = fluid.layers.topk(predict, k=1)
    predict = fluid.layers.reshape(
        predict,
        shape=[data_shape[1], data_shape[2], -1])  # batch_size should be 1
    inference_program = fluid.default_main_program().clone(for_test=True)
    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    assert os.path.exists(args.model_path)
    fluid.io.load_params(exe, args.model_path)
    print("loaded model from: %s" % args.model_path)
    sys.stdout.flush()

    if not os.path.isdir(args.out_path):
        os.makedirs(args.out_path)

    for line in image_list:
        # image_file = args.images_path + "/" + line.strip()
        # filename = os.path.basename(image_file)
        # print(str(cut_path)+"/"+str(line))
        # image = cv2.imread(cut_path+'/'+line)
        #print(11111111,line)
        image = paddle.dataset.image.load_image(
            cut_path+"/" +line, is_color=True).astype("float32")
        image -= IMG_MEAN
        img = paddle.dataset.image.to_chw(image)[np.newaxis, :]
        image_t = fluid.LoDTensor()
        image_t.set(img, place)
        result = exe.run(inference_program,
                         feed={"image": image_t},
                         fetch_list=[predict])
        cv2.imwrite(comb_path + "/" + line + "_result.png",
                    color(result[0]))
    print("predicted images saved in :"+comb_path)
def image_cut(input,output):
    data_list =[]
    path = input
    path_out = output
    img = cv2.imread(path)
    img_exp = np.pad(img, pad_width=((64, 256), (64, 256), (0, 0)), mode="constant", constant_values=(0, 0))
    img_shape = img.shape
    img_exp_shape = img_exp.shape
    h_step = img.shape[0] // 128
    w_step = img.shape[1] // 128
    #print(h_step, w_step)
    h_rest = -(img.shape[0] - 128 * h_step)
    w_rest = -(img.shape[1] - 128 * w_step)
    #print(h_rest, w_rest)
    image_list = []
    for h in range(h_step):
        for w in range(w_step):
            image_sample = img_exp[(h * 128):(h * 128 + 256),
                           (w * 128):(w * 128 + 256), :]
            image_list.append(image_sample)
        # if ori_image[(h * 128):(h * 128 + 256), -256:, :].shape == (256, 256, 3):
        image_list.append(img_exp[(h * 128):(h * 128 + 256), -256:, :])
    for w in range(w_step-1):
        image_list.append(img_exp[-256:, (w * 128):(w * 128 + 256), :])
    image_list.append(img_exp[-256:, -256:, :])
    for i in range(len(image_list)):
        cv2.imwrite(path_out + '/' + str(i) + '.png', image_list[i])
        data_list.append(str(i) + '.png')
    print("cut images saved in :" + path_out)
    return h_step,w_step,h_rest,w_rest,img_shape,img_exp_shape,data_list



def image_comb(h_step,w_step,h_rest,w_rest,img_shape,img_exp_shape,outname,inpath):
    path = inpath
    ori_image = img_exp_shape
    predict_list = []
    files = os.listdir(path)
    for file in range(len(files)):
        file_path = path + '/' + str(file) + '.png'+"_result.png"
        im = cv2.imread(file_path)
        # predict_list.append(im[:, :, :])
        predict_list.append(im[:, :, :])
    # predict_list.append(ori_image[-256:, -256:, :])
    count_temp = 0
    tmp = np.ones([ori_image[0] - 128, ori_image[1] - 128, ori_image[2]])
    for h in range(h_step):
        for w in range(w_step):
            #print(count_temp, predict_list[count_temp][64:-64, 64:-64, :].shape)
            tmp[h * 128:(h + 1) * 128, w * 128:(w + 1) * 128, :] = predict_list[count_temp][64:64 + 128, 64:64 + 128, :]
            count_temp += 1
        tmp[h * 128:(h + 1) * 128, w_rest:, :] = predict_list[count_temp][64:64 + 128, w_rest:, :]
        count_temp += 1
    for w in range(w_step - 1):
        tmp[h_rest:, (w * 128):(w * 128 + 128), :] = predict_list[count_temp][h_rest:, 64:64 + 128, :]
        count_temp += 1
    tmp[h_rest:, w_rest:, :] = predict_list[count_temp][h_rest:, w_rest:, :]
    # print(tmp.shape)
    print("combined image saved in :" + outname)
    cv2.imwrite(outname, tmp[:img_shape[0],:img_shape[1],:img_shape[2]])


def main():
    args = parser.parse_args()
    print_arguments(args)
    in_path = args.images_path
    print('cutting......')
    # in_path = 'input.png'
    cut_path = 'cut_map'
    comb_path = 'comb_map'
    outname = in_path + '_predict.png'
    h_step,w_step,h_rest,w_rest,img_shape,img_exp_shape,data_list = image_cut(in_path,cut_path)
    check_gpu(args.use_gpu)
    print('predicting......')
    infer(args,cut_path,data_list,comb_path)
    print('combining......')
    image_comb(h_step,w_step,h_rest,w_rest,img_shape,img_exp_shape,outname,comb_path)

if __name__ == "__main__":
    main()
