import torch
import cv2
import argparse

from models.yolov1_vgg11 import load_base_model, load_yolo_vgg11
from utils import detect, draw_boxes
from config import S, B, C

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input image',
    default='inference_data/image_1.jpg'
)
parser.add_argument(
    '-t', '--threshold', help='confidence threshold to filter detected boxes',
    default=0.25, type=float
)
args = vars(parser.parse_args())

# Load model and weights.
base_model = load_base_model(pretrained=False)
model = load_yolo_vgg11(base_model, C=C, S=S, B=B)
print('Loading trained YOLO model weights...\n')
checkpoint = torch.load('best.pth')
model.load_state_dict(checkpoint)
model.eval()

# Read and prepare image.
image = cv2.imread(args['input'])
orig_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection.
nms_boxes = detect(model, image, args['threshold'], S=S)
print(nms_boxes)

result = draw_boxes(image, nms_boxes)
cv2.imshow('Result', result)
cv2.waitKey(0)