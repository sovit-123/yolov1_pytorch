import torch
import numpy as np

from yolov1_vgg11 import load_base_model, load_yolo_vgg11
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
    NUM_WORKERS
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

base_model = load_base_model(pretrained=True)
model = load_yolo_vgg11(base_model, C=C, S=S, B=B).to(device)
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
    file='2007_train_labels.txt',
    train=True, 
    transform=get_tensor_transform(),
    S=S, C=C, B=B
)
valid_dataset = DetectionDataset(
    image_size=IMAGE_SIZE, 
    file='2007_val_labels.txt',
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

    scheduler_up = MultiStepLR(
        optimizer, 
        [5, 10, 15], 
        gamma=2.25, verbose=True
    )

    num_iter = 0
    best_valid_loss = np.inf
    for epoch in range(NUM_EPOCHS):
        
        # print(f"Current learning rate: {LEARNING_RATE}")

        # Training.
        train(
            model, train_loader, criterion, 
            optimizer, epoch, NUM_EPOCHS, device
        )

        # Validation.
        validation_loss = validate(model, valid_loader, criterion, device)

        if best_valid_loss > validation_loss:
            best_valid_loss = validation_loss
            print(f"\nNew best validation loss: {best_valid_loss}")
            print('Saving best model...')
            torch.save(model.state_dict(),'best.pth')   
        print(f"Saving model for epoch {epoch+1}")
        torch.save(model.state_dict(),'last.pth')
        scheduler_up.step()
        
