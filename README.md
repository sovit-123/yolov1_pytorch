# YOLOV1 Training Pipeline on the PASCAL VOC Data using the PyTorch Framework



This repository contains YOLOV1 training pipeline on the PASCAL VOC 2007 and 2012 data using the PyTorch framework. **A few points:**



## ***Note: Use this repository/code for any project/learning. A simple reference/attribution to the repository is enough. Completely OPEN for any use.*** 



## Steps to Train

### First of All, Download and Extract the Data (Download to any directory of your choice)

* [Download VOC 2007 trainval from here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar).
  * Or `wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar` in terminal.

* [Download VOC 2007 test from here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar).
  * Or `wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar` in terminal.
* [Download the VOC 2012 trainval from here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
  * Or `wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar`



Execute the following commands in the directory where the downloaded `.tar` files are present.

```
tar xf VOCtrainval_06-Nov-2007.tar
```

```
tar xf VOCtest_06-Nov-2007.tar 
```

```
tar xf VOCtrainval_11-May-2012.tar
```



### Prepare the Text Files and Labels

* Then execute `prepare_data.py` with the correct path argument to the `VOCdevkit` parent directory. Simply, the next folder after the path in the arugment should be `VOCdevkit` The following is an example:

  ```
  python prepare_data.py --path my_pacal_voc_data
  ```

  In the above command `my_pacal_voc_data` should contain the `VOCdevkit` directory for the dataset.

* Then execute `prepare_text_labels.py` with correct command line argument paths to the VOC 2007  and 2012  `Annotations` directory. See [`text_labels.py`](https://github.com/sovit-123/yolov1_pytorch_voc07/blob/main/prepare_text_labels.py) to know what the exact path should be. Example command:

  ```
  python prepare_text_labels.py --annotations-2007 VOCdevkit/VOC2007/Annotations/ --annotations-2012 VOCdevkit/VOC2012/Annotations/
  ```

  

### Train

* Execute `python train.py`. 



## References

* [YOLOv1 paper](https://arxiv.org/pdf/1506.02640v5.pdf): https://arxiv.org/pdf/1506.02640v5.pdf
* Some dataset preparation ideas borrowed from: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/dataset.py
