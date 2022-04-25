import torch
import cv2
import argparse
import pathlib
import os
import time

from models.yolov1_vgg11 import load_base_model, load_yolo_vgg11
from utils import detect, draw_boxes
from config import S, B, C

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='inference_data/video_1.mp4'
)
parser.add_argument(
    '-t', '--threshold', help='confidence threshold to filter detected boxes',
    default=0.25, type=float
)
args = vars(parser.parse_args())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model and weights.
base_model = load_base_model(pretrained=False)
model = load_yolo_vgg11(base_model, C=C, S=S, B=B)
print('Loading trained YOLO model weights...\n')
checkpoint = torch.load('best.pth')
model.load_state_dict(checkpoint)
model.to(device).eval()

# Read the video input.
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# Get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# String name with which to save the resulting video.
save_name = str(pathlib.Path(
    args['input']
)).split(os.path.sep)[-1].split('.')[0]
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second. 

# Read until end of video.
while(cap.isOpened()):
    # Capture each frame of the video.
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        # Run detection.
        nms_boxes = detect(
            model, image, args['threshold'], 
            S=S, device=device
        )
        end_time = time.time()

        result = draw_boxes(image, nms_boxes)
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1

        cv2.putText(result, f"{fps:.1f} FPS", 
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 
                    2, lineType=cv2.LINE_AA)

        cv2.imshow('Result', result)
        out.write(result)
        # Press `q` to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release VideoCapture() object.
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()