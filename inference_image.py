import torch
import cv2
import argparse

from utils import detect, draw_boxes
from config import S, B, C
from models.create_model import create_model

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input image',
    default='inference_data/image_1.jpg'
)
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
args = vars(parser.parse_args())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model and weights.
create_model = create_model[args['model']]
model = create_model(C, S, B, pretrained=False).to(device)
print('Loading trained YOLO model weights...\n')
checkpoint = torch.load(args['weights'])
model.load_state_dict(checkpoint)
model.to(device).eval()

# Read and prepare image.
image = cv2.imread(args['input'])
# image = cv2.resize(image, (448, 448))
orig_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection.
nms_boxes, scores = detect(model, image, args['threshold'], S=S, device=device)
# print(nms_boxes)

result = draw_boxes(image, nms_boxes)
cv2.imshow('Result', result)
cv2.waitKey(0)