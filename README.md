# yolov5-tensorrt-python
不依赖于pytorch,只用tensorrt和numpy进行加速,在1080ti上测试达到了160fps

1.需要安装tensorrt python版

2.安装pycuda

3.将训练好的模型（这里使用的是yolov5-4.0训练的）按照https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v4.0/yolov5 上的方法转成libmyplugins.so和yolov5s.engine文件

4.修改main.py中的PLUGIN_LIBRARY和engine_file_path路径就可以使用
