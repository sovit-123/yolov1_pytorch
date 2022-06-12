import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOLoss(nn.Module):
    """
    YOLOv1 loss.
    Paper: https://arxiv.org/pdf/1506.02640v5.pdf
    """

    def __init__(self, S=7, B=2, C=20):
        super(YOLOLoss, self).__init__()
        
        # ========================== #
        # S = Number of grid rows and columns an image is divided into.
        # B = Number of bounding boxes that the model predicts.
        # C = Number of classes in the dataset.
        # ========================== #

        self.S = S
        self.B = B
        self.C = C

        # To decrease loss from confidence predictions for boxes
        # that don't contain object.
        self.lambda_noobj = 0.5 
        # Increase loss from bounding box coordinate predictions.
        self.lambda_coord = 5

        # MSE for calculation later on.
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, preds, targets):
        """
        :param preds: Predictions made by the model. 
                      Shape = (BATCH_SIZE, S*S(C+B*5)). 
        :param target: Ground truth labels.
        """

        # If S=7, B=2, C=20, `preds` shape will be (BATCH_SIZE, 1470).
        # Reshape them to (BATCH_SIZE, 7, 7, 30).
        preds = preds.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # As per paper, we need to consider the box having highest IoU out of
        # the two predicted boxes.
        box1_iou = intersection_over_union(preds[..., 21:25], targets[..., 21:25])
        box2_iou = intersection_over_union(preds[..., 26:30], targets[..., 21:25])
        # Concatenate IoUs for later comparison.
        all_ious = torch.cat([box1_iou.unsqueeze(0), box2_iou.unsqueeze(0)], dim=0)
        # Extract the best IoU value and indices. The indices can either be 
        # 0 or 1.
        max_ious, best_index = torch.max(all_ious, dim=0)
        # Index position 20 in the ground truth vector indicates whether
        # an object is present or not for that grid. If present, the value is
        # 1, else it is 0.
        gt_box_present = targets[..., 20].unsqueeze(3)

        # ========================== #
        # Handling Box Coordinates   #
        # ========================== #
        # Box predictions shape will be (BATCH_SIZE, 7, 7, 4) where the 
        # last 4 values will be zeros or bbox predictions depending on 
        # whether the ground truth contains an object or not.
        box_predictions = gt_box_present * (
            (
                best_index * preds[..., 26:30]
                + (1 - best_index) * preds[..., 21:25]
            )
        )
        box_targets = gt_box_present * targets[..., 21:25]
        # Square root of width and height to handle deviations 
        # for large and small boxes.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) \
            * torch.sqrt(
                torch.abs(box_predictions[..., 2:4] + 1e-6)
            )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        # `box_loss` will be used in final loss calculation.
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # ========================== #
        # Handling Object Loss       #
        # ========================== #
        # `pred_conf` is the confidence score for bbox with the highest IoU.
        pred_conf = (
            best_index * preds[..., 25:26] + \
                (1 - best_index) * preds[..., 20:21]
        )
        object_present_loss = self.mse(
            torch.flatten(gt_box_present * pred_conf),
            torch.flatten(gt_box_present * targets[..., 20:21]),
        )

        # ========================== #
        # Handling No Object Loss    #
        # ========================== #
        no_object_loss = self.mse(
            torch.flatten((1 - gt_box_present) * preds[..., 20:21], start_dim=1),
            torch.flatten((1 - gt_box_present) * targets[..., 20:21], start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten((1 - gt_box_present) * preds[..., 25:26], start_dim=1),
            torch.flatten((1 - gt_box_present) * targets[..., 20:21], start_dim=1)
        )

        # ========================== #
        # Handling Class Loss        #
        # ========================== #
        class_loss = self.mse(
            torch.flatten(gt_box_present * preds[..., :20], end_dim=-2,),
            torch.flatten(gt_box_present * targets[..., :20], end_dim=-2,),
        )

        # ========================== #
        # Calculate Final Loss       #
        # ========================== #
        loss = (
            self.lambda_coord * box_loss 
            + object_present_loss 
            + self.lambda_noobj * no_object_loss  
            + class_loss
        )
        return loss

if __name__ == '__main__':    
    t = torch.tensor([
        [200, 400, 300, 500],
        [300, 500, 500, 350]
    ])
    b1 = torch.tensor([[0, 0, 0, 0]])
    b2 = torch.tensor([[200, 390, 300, 500]])

    iou1 = intersection_over_union(b1, t)
    iou2 = intersection_over_union(b2, t)
    print('IoU of first box:\n', iou1)
    print('IoU of second box:\n', iou2)

    ious = torch.cat([iou1.unsqueeze(0), iou2.unsqueeze(0)], dim=0)
    print('Concatenated IoUs:\n', ious)
    max_ious, best_bbox = torch.max(ious, dim=0)
    print('Max IoU: ', max_ious) 
    print('Max IoU index:', best_bbox)

    rand_tensor_1 = torch.rand([4, 7, 7, 30])
    rand_tensor_2 = torch.rand([4, 7, 7, 30])
    rand_1_reshaped = rand_tensor_1.view(4, 1470)
    yolo_loss = YOLOLoss()
    loss = yolo_loss(rand_1_reshaped, rand_tensor_1)
    print(f"YOLO loss test => YOLO loss = {loss}")