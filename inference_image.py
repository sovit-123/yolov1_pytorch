import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from yolov1_vgg11 import load_base_model, load_yolo_vgg11
from utils import cellboxes_to_boxes, non_max_suppression
from config import S

# Load model and weights.
base_model = load_base_model(pretrained=False)
model = load_yolo_vgg11(base_model, C=20, S=7, B=2)
checkpoint = torch.load('best.pth')
model.load_state_dict(checkpoint)

# Read and prepare image.
image = cv2.imread('inference_data/image_2.jpg')
orig_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (448, 448))
image = np.transpose(image, (2, 0, 1))
image = torch.tensor(image, dtype=torch.float32)
image = torch.unsqueeze(image, axis=0)

model.eval()
outputs = model(image)
print(outputs.shape)

threshold = 0.95

bboxes = cellboxes_to_boxes(outputs, S=S)
batch_size = image.shape[0]
all_pred_boxes = []
for idx in range(batch_size):
    nms_boxes = non_max_suppression(
        bboxes[idx],
        iou_threshold=0.5,
        box_format='midpoint',
    )
    # Filter out the boxes based on confidence score.
    nms_boxes = [box for box in nms_boxes if box[1] > threshold]

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