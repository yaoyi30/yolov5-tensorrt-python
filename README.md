# yolov5-tensorrt-python
不依赖于pytorch,只用tensorrt和numpy进行加速,在1080ti上测试达到了160fps

1.需要安装tensorrt python版

2.安装pycuda

3.将训练好的模型（这里使用的是yolov5-4.0训练的s模型）按照https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v4.0/yolov5 上的方法转成libmyplugins.so和yolov5s.engine文件

4.修改dector_trt.py中categories里面的类别为自己的类别

5.修改main.py中的PLUGIN_LIBRARY和engine_file_path路径就可以使用

参考：

1.https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v4.0/yolov5

2.https://github.com/ultralytics/yolov5/tree/v4.0

3.https://github.com/cong/yolov5_deepsort_tensorrt

4.https://gitee.com/chaucerg/yolov5-tensorrt
