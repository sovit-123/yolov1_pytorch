# A Simple YOLOV1 Training Pipeline on the PASCAL VOC 2007 Data using the PyTorch Framework



This repository contains YOLOV1 training pipeline on the PASCAL VOC 2007 data using the PyTorch framework. **A few points:**

* This respository can be used for understanding the entire YOLOV1 training pipeline, starting from:
  * Dataset creation.
  * Model building.
  * YOLO loss function.
  * Evaluation.
  * Prediction/inference on images and videos using the trained model.
* The code base contains the YOLOV1 backbone (termed as Darknet here onward) and VGG11 backbones as they are the simplest to understand.
* The code base will be well commented for proper understanding of each part.



***Note: Use this repository/code for any project/learning with/without attribution. Completely OPEN for any use.*** 



## First of All, Download and Extract the Data

* [Download VOC 2007 trainval from here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar).
* [Download VOC 2007 test from here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar).

Execute the following commands in the directory where the downloaded `.tar` files are present.

```
tar xvf VOCtrainval_06-Nov-2007.tar
```

```
tar xvf VOCtest_06-Nov-2007.tar 
```

* Then execute `prepare_data.py` with the correct path argument to the `VOCdevkit` parent directory. The following is an example:

```
python prepare_data.py --path ../xml_od_data/pascal_voc_original/
```

* Then execute `prepare_text_labels.py`.



## References

* [YOLOv1 paper](https://arxiv.org/pdf/1506.02640v5.pdf): https://arxiv.org/pdf/1506.02640v5.pdf
* Some dataset preparation ideas borrowed from: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/dataset.py
# yolov1_pytorch_voc07
# yolov1_pytorch_voc07
