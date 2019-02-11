# mask_rcnn代码，源代码地址：https://github.com/matterport/Mask_RCNN
# windows平台代码运行
  
1、在Mask_RCNN根目录下执行：python setup.py install，
执行完后会看到目录下多了些文件
![Image text](https://github.com/taogougou/img_folder/blob/master/mask_rcnn_img1.png?raw=true)
    
2、下载预训练过的COCO数据集权重mask_rcnn_balloon.h5，放到项目下
地址：https://github.com/matterport/Mask_RCNN/releases

  
3、如果需要在COCO数据集上训练或测试，需要安装pycocotools，
地址：https://github.com/philferriere/cocoapi   解压后，文件：cocoapi-master
安装：进入PythonAPI目录，执行命令：
install pycocotools locally
python setup.py build_ext --inplace
install pycocotools to the Python site-packages
python setup.py build_ext install
如果报错：error: Unable to find vcvarsall.bat 说明没有安装Microsoft Visual C++ 2015
这里下载安装，安装完后重启电脑生效，已经下载，文件为：visualcppbuildtools_full.exe，
地址：https://blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat/

![Image text](https://github.com/taogougou/img_folder/blob/master/mask_rcnn_img2.png?raw=true)
4、	将把pycocotools里面所有文件复制到项目下  
![Image text](https://github.com/taogougou/img_folder/blob/master/mask_rcnn_img3.png?raw=true)
5、运行samples/coco/inspect_data.ipynb试例代码的时候会出现如下的错误： 
![Image text](https://github.com/taogougou/img_folder/blob/master/mask_rcnn_img4.png?raw=true)
是因为没有安装imgaug，安装方法，官网有详细介绍，可以三种方法按照：地址：
https://github.com/aleju/imgaug
![Image text](https://github.com/taogougou/img_folder/blob/master/mask_rcnn_img5.png?raw=true)
在安装imgaug时候，会报以下错：
![Image text](https://github.com/taogougou/img_folder/blob/master/mask_rcnn_img6.png?raw=true)
 这个查找了好多资料，都没有解决，突然发现官网其实已经给出了解决方法，注意要提前将相关的包都安装好后再安装imgaug
 ![Image text](https://github.com/taogougou/img_folder/blob/master/mask_rcnn_img7.png?raw=true)
 将以上包都安装好后，在重新安装，我选择的是第二种安装方式：
pip install git+https://github.com/aleju/imgaug
当然提前要安装好git工具，负责无法成功，显示一下信息表示安装成功：
 ![Image text](https://github.com/taogougou/img_folder/blob/master/mask_rcnn_img8.png?raw=true)




