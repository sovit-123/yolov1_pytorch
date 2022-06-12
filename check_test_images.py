"""
To check results for the test images (2007_test.txt) file using the best model.
"""

import torch
import cv2
import argparse

from utils import detect, draw_boxes
from config import S, B, C
from models.create_model import create_model

parser = argparse.ArgumentParser()
parser.add_argument(
    '-t', '--threshold', help='confidence threshold to filter detected boxes',
    default=0.25, type=float
)
parser.add_argument(
    '-m', '--model', default='yolov1_vgg11', 
    help='the model to train with, see models/create_model.py for all \
          available models'
)
parser.add_argument(
    '-w', '--weights', default='best.pth', 
    help='path to model weight'
)
parser.add_argument(
    '-d', '--device', 
    default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    help='computing device'
)
args = vars(parser.parse_args())

device = args['device']

# Load model and weights.
create_model = create_model[args['model']]
model = create_model(C, S, B, pretrained=False).to(device)
print('Loading trained YOLO model weights...\n')
checkpoint = torch.load(args['weights'], map_location=args['device'])
model.load_state_dict(checkpoint)
model.to(device).eval()

with open('2007_test.txt', 'r') as f:
    test_image_paths = f.readlines()
f.close()

for image_path in test_image_paths:
    # Read and prepare image.
    image = cv2.imread(f"{image_path[0:-1]}")
    # image = cv2.resize(image, (448, 448))
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Run detection.
    nms_boxes, scores = detect(model, image, args['threshold'], S=S, device=device)
    # print(nms_boxes)
    
    result = draw_boxes(image, nms_boxes)
    cv2.imshow('Result', result)
    cv2.waitKey(0)