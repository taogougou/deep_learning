# mask_rcnn代码，源代码地址：https://github.com/matterport/Mask_RCNN
# windows平台代码运行
  
1、二、	在Mask_RCNN根目录下执行：python setup.py install，
执行完后会看到目录下多了些文件
![Image text](https://github.com/taogougou/img_folder/blob/master/mask_rcnn_img1.png?raw=true)
    
2、三、	下载预训练过的COCO数据集权重mask_rcnn_balloon.h5，放到项目下
地址：https://github.com/matterport/Mask_RCNN/releases

  
3、四、	如果需要在COCO数据集上训练或测试，需要安装pycocotools，
地址：https://github.com/philferriere/cocoapi   解压后，文件：cocoapi-master
安装：进入PythonAPI目录，执行命令：
# install pycocotools locally
python setup.py build_ext --inplace
# install pycocotools to the Python site-packages
python setup.py build_ext install
如果报错：error: Unable to find vcvarsall.bat 说明没有安装Microsoft Visual C++ 2015
这里下载安装，安装完后重启电脑生效，已经下载，文件为：visualcppbuildtools_full.exe，地址：https://blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat/

![Image text](https://github.com/taogougou/img_folder/blob/master/mask_rcnn_img2.png?raw=true)
  




