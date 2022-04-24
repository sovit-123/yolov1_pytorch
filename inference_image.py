import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

from models.yolov1_vgg11 import load_base_model, load_yolo_vgg11
from utils import detect
from config import S, B, C

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input image',
    default='inference_data/image_2.jpg'
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

# Read and prepare image.
image = cv2.imread(args['input'])
orig_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model.eval()

nms_boxes = detect(model, image, args['threshold'], S=S)

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Rectangle potch
    for box in boxes:
        print(box)
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

plot_image(orig_image, nms_boxes)