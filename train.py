import torch
import numpy as np
import argparse

from models.create_model import create_model
from loss import YOLOLoss
from dataset import DetectionDataset
from transforms import get_tensor_transform
from torch.utils.data import DataLoader
from training_utils import train, validate
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from config import (
    S, B, C,
    PRETRAINED
)
from utils import plot_loss

parser = argparse.ArgumentParser()
parser.add_argument(
    '-w', '--weights', default=None,
    help='path to weights if wanting to resume training'
)
parser.add_argument(
    '-m', '--model', default='yolov1_vgg11', 
    help='the model to train with, see models/create_model.py for all \
          available models'
)
parser.add_argument(
    '-e', '--epochs', default=135, type=int,
    help='number of epochs to train for'
)
parser.add_argument(
    '-b', '--batch-size', dest='batch_size', default=4, type=int,
    help='batch size for data loader'
)
parser.add_argument(
    '-j', '--workers', default=8, type=int,
    help='parallel workers for data loaders'
)
parser.add_argument(
    '-s', '--image-size', dest='image_size', default=448,
    help='image size to resize to during data loading'
)
parser.add_argument(
    '-lr', '--learning-rate', dest='learning_rate', default=0.0001, type=float,
    help='default learning rate for optimizer'
)
parser.add_argument(
    '-pt', '--pretrained', action='store_true', 
    help='whether to use pretrained model or not'
)
args = vars(parser.parse_args())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

create_model = create_model[args['model']]
model = create_model(C, S, B, pretrained=args['pretrained']).to(device)
if args['weights'] is not None:
    print('Loading weights to resume training...')
    checkpoint = torch.load(args['weights'], map_location=device)
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
# Different learning rate for the features and head.
params = []
param_dict = dict(model.named_parameters())
for key, value in param_dict.items():
    if key.startswith('features'):
        params += [{'params':[value], 'lr':args['learning_rate']}]
    else:
        params += [{'params':[value], 'lr':args['learning_rate']*1}]
# optimizer = torch.optim.SGD(
#     params, lr=LEARNING_RATE,
#     momentum=0.9, weight_decay=0.0005
# )
optimizer = torch.optim.SGD(
    model.parameters(), lr=args['learning_rate'],
    momentum=0.9, weight_decay=0.0005
)
# optimizer = torch.optim.Adam(
#     params, lr=LEARNING_RATE,
# )

train_dataset = DetectionDataset(
    image_size=args['image_size'], 
    file='train_labels.txt',
    train=True, 
    transform=get_tensor_transform(),
    S=S, C=C, B=B
)
valid_dataset = DetectionDataset(
    image_size=args['image_size'], 
    file='2007_test_labels.txt',
    train=False, 
    transform=get_tensor_transform(),
    S=S, C=C, B=B
)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")

if __name__ == '__main__':
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args['batch_size'],
        shuffle=True, num_workers=args['workers']
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=args['batch_size'],
        shuffle=False, num_workers=args['workers']
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
        gamma=1.047128548, verbose=False
    )
    # To decrease the learning.
    scheduler_down = MultiStepLR(
        optimizer,
        [160, 195],
        gamma=0.1, verbose=False
    )

    num_iter = 0
    best_valid_loss = np.inf
    train_loss, valid_loss = [], []
    for epoch in range(args['epochs']):
        
        # Training.
        training_loss = train(
            model, train_loader, criterion, 
            optimizer, epoch, args['epochs'], device
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
        # scheduler_up.step()
        # scheduler_down.step()

        plot_loss(train_loss, valid_loss)
    torch.save(model.state_dict(),'last.pth')