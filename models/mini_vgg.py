import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

# The VGG11 model.
class MiniVGG11(nn.Module):
    def __init__(self, in_channels, num_classes=1000, pretrained=False):
        super(MiniVGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # Convolutional layers 
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # Fully connected linear layers.
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

        if not pretrained:
            self.weights_init()

    def weights_init(self):
        print('INITIALIZING RANDOM WEIGHTS...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.avgpool(x)
    #     # flatten to prepare for the fully connected layers
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     return x

class YOLOVGG11(nn.Module):
    def __init__(self, base_model, C=20, S=7, B=2):
        super(YOLOVGG11, self).__init__()
        self.features = base_model.features

        # YOLO Head according to the paper.
        # The authors add four Conv2D layers to the classification
        # backbone, followed by two fully connected layers.
        self.yolo_head = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, S * S * (C + B * 5))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.yolo_head(x)
        return x

def load_base_model(pretrained=False):
    vgg11 = MiniVGG11(3, pretrained=pretrained)
    if not pretrained:
        print('Not loading pretrained weights...')
    if pretrained:
        print('Loading pretrained weights...')
        vgg11.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/vgg11_bn-6002323d.pth'
        ))
    return vgg11

def load_yolo_vgg11(base_model, C, S, B):
    yolo_vgg11 = YOLOVGG11(base_model, C, S, B)
    return yolo_vgg11

if __name__ == '__main__':
    vgg11 = load_base_model(pretrained=False)
    yolo_vgg11 = load_yolo_vgg11(vgg11)
    print(yolo_vgg11)

    x = torch.rand([1, 3, 448, 448])
    print(f"Dummy input tensor shape: {x.shape}")
    out = yolo_vgg11(x)
    print(f"Output shape: {out.shape}")

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in yolo_vgg11.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in yolo_vgg11.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")