from models import *

def return_yolov1(C, S, B, pretrained=True):
    print('Loading YOLOv1 Darknet model...')
    base_model = yolov1.load_base_model(pretrained=pretrained)
    model = yolov1.load_yolo_model(base_model, C=C, S=S, B=B)
    return model

def return_mini_vgg(C, S, B, pretrained=True):
    print('Loading Mini VGG model...')
    base_model = yolov1_mini_vgg.load_base_model(pretrained=pretrained)
    model = yolov1_mini_vgg.load_yolo_model(base_model, C=C, S=S, B=B)
    return model

def return_yolov1_vgg11(C, S, B, pretrained=True):
    print('YOLOV1 with VGG11 backbone...')
    base_model = yolov1_vgg11.load_base_model(pretrained=pretrained)
    model = yolov1_vgg11.load_yolo_model(base_model, C=C, S=S, B=B)
    return model

create_model = {
    'yolov1': return_yolov1,
    'yolov1_mini_vgg': return_mini_vgg,
    'yolov1_vgg11': return_yolov1_vgg11,
}