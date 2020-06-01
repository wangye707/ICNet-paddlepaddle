# ICNet-paddlepaddle
使用ICNet模型对航拍图片(遥感图像)进行图像分割

本次数据集提供以及部分参考代码提供方：https://github.com/ximimiao/deeplabv3-Tensorflow

百度网盘下载数据集地址：https://github.com/ximimiao/deeplabv3-Tensorflow
密码：https://github.com/ximimiao/deeplabv3-Tensorflow
已经训练完成模型下载地址：https://github.com/ximimiao/deeplabv3-Tensorflow

训练环境依赖：
numpy == 1.18.1
opencv-python == 4.2.0.32
paddlepaddle-gpu == 1.7.1.post97
pandas == 0.25.3

执行方式：
1.图片裁剪
python preprocess.py
2.训练神经网络
简化版本，均使用默认参数请执行：python train.py 
指定参数请执行：python train.py  --batch_size=64 --checkpoint_path=chkpnt --init_model=chkpnt/1000 --use_gpu=True
3.评估网络模型
python eval.py --chkpnt/3000

4.通过已训练完成的网络模型预测图片
直接预测：python infer1.py --model_path=chlpnt/370000 --images_path=input.png
膨胀预测优化：python infer_exp.py --model_path=chlpnt/370000 --images_path=input.png

迭代过程展示（部分细节）：
