import torch
import numpy as np
import argparse

from models.yolov1_vgg11 import load_base_model, load_yolo_vgg11
from loss import YOLOLoss
from dataset import DetectionDataset
from transforms import get_tensor_transform
from torch.utils.data import DataLoader
from training_utils import train, validate
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from config import (
    S, B, C,
    NUM_EPOCHS, 
    BATCH_SIZE,
    LEARNING_RATE, 
    IMAGE_SIZE, 
    NUM_WORKERS,
    PRETRAINED
)
from utils import plot_loss

parser = argparse.ArgumentParser()
parser.add_argument(
    '-w', '--weights', default=None,
    help='path to weights if wanting to resume training'
)
args = vars(parser.parse_args())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

base_model = load_base_model(pretrained=PRETRAINED)
model = load_yolo_vgg11(base_model, C=C, S=S, B=B).to(device)
if args['weights'] is not None:
    print('Loading weights to resume training...')
    checkpoint = torch.load(args['weights'])
    model.load_state_dict(checkpoint)
print(model)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Loss and optimizer.
criterion = YOLOLoss(S=S, B=B)
optimizer = torch.optim.SGD(
    model.parameters(), lr=LEARNING_RATE,
    momentum=0.9, weight_decay=0.0005
)

train_dataset = DetectionDataset(
    image_size=IMAGE_SIZE, 
    file='train_labels.txt',
    train=True, 
    transform=get_tensor_transform(),
    S=S, C=C, B=B
)
valid_dataset = DetectionDataset(
    image_size=IMAGE_SIZE, 
    file='2007_test_labels.txt',
    train=False, 
    transform=get_tensor_transform(),
    S=S, C=C, B=B
)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")

if __name__ == '__main__':
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    # To increase the learning rate slowly till 50 epochs.
    scheduler_up = MultiStepLR(
        optimizer, 
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50
        ], 
        gamma=1.047128548, verbose=True
    )
    # To decrease the learning rate at 75 and 105 epochs.
    scheduler_down = MultiStepLR(
        optimizer,
        [160, 195],
        gamma=0.1, verbose=True
    )

    num_iter = 0
    best_valid_loss = np.inf
    train_loss, valid_loss = [], []
    for epoch in range(NUM_EPOCHS):
        
        # Training.
        training_loss = train(
            model, train_loader, criterion, 
            optimizer, epoch, NUM_EPOCHS, device
        )

        # Validation.
        validation_loss = validate(model, valid_loader, criterion, device)

        train_loss.append(training_loss)
        valid_loss.append(validation_loss)

        if best_valid_loss > validation_loss:
            best_valid_loss = validation_loss
            print(f"\nNew best validation loss: {best_valid_loss}")
            print('Saving best model...')
            torch.save(model.state_dict(),'best.pth')   
        print(f"Saving model for epoch {epoch+1}\n")
        torch.save(model.state_dict(),'last.pth')
        scheduler_up.step()
        # scheduler_down.step()

        plot_loss(train_loss, valid_loss)