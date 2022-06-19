import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from config import S

plt.style.use('ggplot')

def detect(model, image, threshold, S, device):
    """
    Run detection for inference.

    :param model: The neural network model.
    :param image: Image NumPy array in RGB format.
    :param threshold: Detection threshold for filtering boxes.
    :param S: Grid size, default S = 7.
    :param device: The computation device (GPU or CPU).

    Returns:
        nms_boxes: Detected boxes after Non-Max Suppression.
        score_list: The score list corresponding to the final `nms_boxes`.
    """
    corner_list = [] # List to store coordinates in x1, x2, y1, y2 format.
    score_list = [] # List to store corresponding scores.
    class_list = [] # List to store corresponding classes.
    orig_image = image.copy()
    height, width, _ = orig_image.shape
    image = cv2.resize(image, (448, 448))
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32)/255.
    image = torch.unsqueeze(image, axis=0)
    outputs = model(image.to(device))
    bboxes = cellboxes_to_boxes(outputs, S=S)

    for i, bbox in enumerate(bboxes[0]):
        x1, y1, x2, y2 = yolo2bbox(bbox[2:], width, height)
        # Check that all coordinates are > 0 and score > threshold.
        if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0 and bbox[1] > threshold:
            corner_list.append([x1, y1, x2, y2])
            score_list.append(bbox[1])
            class_list.append(bbox[0])
    if len(corner_list) > 0:
        nms_indices = torchvision.ops.nms(
            torch.tensor(corner_list), 
            torch.tensor(score_list),
            iou_threshold=0.5
        )
        nms_boxes = [corner_list[i] for i in nms_indices]
        final_scores = [score_list[i] for i in nms_indices]
        final_classes=  [class_list[i] for i in nms_indices]
        return nms_boxes, final_scores, final_classes
    else:
        return [], [], []

def intersection_over_union(
    boxes_preds, boxes_labels, 
    box_format="midpoint",
    epsilon=1e-6
):
    """
    Calculates intersection over union for bounding boxes.
    
    :param boxes_preds (tensor): Bounding box predictions of shape (BATCH_SIZE, 4)
    :param boxes_labels (tensor): Ground truth bounding box of shape (BATCH_SIZE, 4)
    :param box_format (str): midpoint/corners, if boxes (x,y,w,h) format or (x1,y1,x2,y2) format
    :param epsilon: Small value to prevent division by zero.
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4] 
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = (box1_area + box2_area - intersection + epsilon)

    return intersection / union

# Borrowed/adapted from:
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/utils.py
def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds

# Borrowed/adapted from:
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/utils.py
def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out, S=S).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def plot_loss(train_loss, valid_loss, out_dir):
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    ax.plot(
        valid_loss, color='tab:orange', linestyle='-', 
        label='valid loss'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    figure.savefig(os.path.join(out_dir, 'loss.png'))

def draw_boxes(image, boxes, class_labels, class_names, colors):
    """
    Draw bounding boxes around an image.

    :param image: NumPy array image in RGB format.
    :param boxes: NMS appied bounding boxes. Shape is [N, 6].
        Normalized box coordinates start from index 2 in the format of
        [x_center, y_center, normalized width, normalized height].

    Returns:
        image: NumPy array image with bounding boxes drawn.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Get the original image height and width.
    for i, box in enumerate(boxes):
        color = colors[int(class_labels[i])][::-1]
        class_name = class_names[int(class_labels[i])]
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(
            image,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            color,
            1, cv2.LINE_AA
        )
        cv2.putText(
            image,
            str(class_name),
            (int(x_min), int(y_min)-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, color=color,
            thickness=1,
            lineType=cv2.LINE_AA
        )
    return image

def yolo2bbox(bboxes, width, height):
    """
    Function to convert bounding boxes in YOLO format to 
    xmin, ymin, xmax, ymax.
    
    Parmaeters:
    :param bboxes: Normalized [x_center, y_center, width, height] list
    :param width: Original width of the image.
    :param height: Original height of the image.

    return: xmin, ymin, xmax, ymax relative to original image size.
    """
    xmin, ymin = (bboxes[0]-bboxes[2]/2) * width, (bboxes[1]-bboxes[3]/2) * height
    xmax, ymax = (bboxes[0]+bboxes[2]/2) * width, (bboxes[1]+bboxes[3]/2) * height
    return xmin, ymin, xmax, ymax

def check_valid_loop(
    outputs, 
    images, 
    epoch, 
    i, 
    out_dir, 
    class_names,
    colors
):
    """
    Saves results from the validation loop during the training phase.
    """
    corner_list = [] # List to store coordinates in x1, x2, y1, y2 format.
    score_list = [] # List to store corresponding scores.
    class_list = [] # List to store corresponding classes.
    image = images[0]
    height, width = image.shape[1:]
    threshold = 0.25
    bboxes = cellboxes_to_boxes(outputs, S=S)
    for i, bbox in enumerate(bboxes[0]):
        x1, y1, x2, y2 = yolo2bbox(bbox[2:], width, height)
        # Check that all coordinates are > 0 and score > threshold.
        if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0 and bbox[1] > threshold:
            corner_list.append([x1, y1, x2, y2])
            score_list.append(bbox[1])
            class_list.append(bbox[0])
    if len(corner_list) > 0:
        nms_indices = torchvision.ops.nms(
            torch.tensor(corner_list), 
            torch.tensor(score_list),
            iou_threshold=0.5
        )
        nms_boxes = [corner_list[i] for i in nms_indices]
        final_scores = [score_list[i] for i in nms_indices]
        final_classes=  [class_list[i] for i in nms_indices]
        image_1 = np.array(torch.permute(images[0].cpu(), (1, 2, 0)))
        result = draw_boxes(
            image_1, 
            nms_boxes, 
            final_classes,
            class_names, 
            colors
        )
        cv2.imwrite(
            os.path.join(out_dir, 'valid_results', f"image_e{epoch}_iter{i}.png"), 
            result*255.
        )
        # cv2.imshow('Result', result)
        # cv2.waitKey(0)

def set_training_dir(dir_name=None):
    """
    This functions counts the number of training directories already present
    and creates a new one in `outputs/training/`. 
    And returns the directory path.
    """
    if not os.path.exists('outputs/training'):
        os.makedirs('outputs/training')
    if dir_name:
        new_dir_name = f"outputs/training/{dir_name}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name
    else:
        num_train_dirs_present = len(os.listdir('outputs/training/'))
        next_dir_num = num_train_dirs_present + 1
        new_dir_name = f"outputs/training/res_{next_dir_num}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name

def main():
    boxes_preds = torch.tensor([200, 300, 400, 500])
    boxes_labels = torch.tensor([200, 300, 400, 500])
    boxes_preds = torch.unsqueeze(boxes_preds, axis=0)
    boxes_labels = torch.unsqueeze(boxes_labels, axis=0)
    print(intersection_over_union(boxes_preds, boxes_labels, 'corners'))

if __name__ == '__main__':
    main()