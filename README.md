# ICNet-paddlepaddle

使用ICNet模型对航拍图片(遥感图像)进行图像分割

数据集下载：
* 百度网盘下载数据集地址：https://pan.baidu.com/s/1DkyQvhvHXVf6EMzm6uLZWQ 
* 密码：37nk
* 说明：5张图片中前4张用来切分，然后训练网络模型，最后一张图片5.png用来做测试

已训练完成模型下载（其中有1000步和40万步，1000用来参考，40万是最终训练模型）：
* 百度网盘下载地址：https://pan.baidu.com/s/11nT6_H5MVrPDTkMX5bDK1g
* 密码：qbxd
* 说明：此处大家也可以自行训练，不采用此处提供的模型。如果下载，请放在模型文件夹下（chkpnt/400000）

训练环境关键包依赖
* numpy == 1.18.1
* opencv-python == 4.2.0.32
* paddlepaddle-gpu == 1.7.1.post97
* pandas == 0.25.3

执行方式（注意，以下脚本有生成文件指令，请附带sudo权限）：
```
1.图片裁剪
python preprocess.py 

2.训练神经网络
简化版本，均使用默认参数请执行：
python train.py 
指定参数请执行：              
python train.py  --batch_size=64 --checkpoint_path=chkpnt --init_model=chkpnt/1000 --use_gpu=True
说明：checkpoint_path：  模型将保存的路径，默认为10000步保存一次。
      init_model：      预训练模型的路径，本次没有给出预训练模型。
      模型恢复训练方式：  指定最新的模型路径来恢复训练，记得修改当
                        前迭代步数以保证正常训练（如在1000步时终止
                        ，记得修改迭代步数从1000开始，而不是默认参数0）
     
3.评估网络模型
python eval.py --model_path=chkpnt/400000
说明：model_path是指定的模型文件路径

4.通过已训练完成的网络模型预测图片
直接预测：
python infer1.py --model_path=chkpnt/400000 --images_path=dataset/origin/5.png
膨胀预测优化（推荐使用）：
python infer_exp.py --model_path=chkpnt/400000 --images_path=dataset/origin/5.png
```
迭代过程展示（部分细节）：

<img src="https://github.com/wangye707/ICNet-paddlepaddle/blob/master/1.jpg" width="350" height="350" />

膨胀预测优化：

<img src="https://github.com/wangye707/ICNet-paddlepaddle/blob/master/2.jpg" width="300" height="300" />

最终结果对比图1：

<img src="https://github.com/wangye707/ICNet-paddlepaddle/blob/master/3.jpg" width="400" height="300" />

最终结果对比图2：

<img src="https://github.com/wangye707/ICNet-paddlepaddle/blob/master/4.jpg" width="500" height="300" />

代码部分细节解释请参考我的CSDN博客：https://blog.csdn.net/qq_28626909/article/details/106489285

个人微信联系方式：wy1119744330 (添加请备注来意)

本次数据集提供以及部分数据处理参考代码：https://github.com/ximimiao/deeplabv3-Tensorflow


