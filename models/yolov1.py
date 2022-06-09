import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

class DarkNet(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        num_classes=1000, 
        pretrained=False, 
        initialize_weights=True
    ):
        super(DarkNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = self._create_conv_layers()
        self.pool = self._pool()
        self.fcs = self._create_fc_layers()

        if initialize_weights:
            # random initialization of the weights...
            # ... just like the original paper
            self._initialize_weights()

    def _create_conv_layers(self):
        conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.MaxPool2d(2),

            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        return conv_layers

    # def _create_fc_layers(self):
    #     fc_layers = nn.Sequential(
    #         nn.AvgPool2d(7),
    #         nn.Linear(1024, self.num_classes)
    #     )
    #     return fc_layers

    def _pool(self):
        pool = nn.Sequential(
            nn.AvgPool2d(7),
        )
        return pool
    
    def _create_fc_layers(self):
        fc_layers = nn.Sequential(
            nn.Linear(1024, self.num_classes)
        )
        return fc_layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_in',
                    nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze()
        x = self.fcs(x)
        return x

class YOLODarknet(nn.Module):
    def __init__(self, base_model, C=20, S=7, B=2):
        super(YOLODarknet, self).__init__()
        self.features = base_model.features

        # YOLO Head according to the paper.
        # The authors add four Conv2D layers to the classification
        # backbone, followed by two fully connected layers.
        self.yolo_head = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, S * S * (C + B * 5))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.yolo_head(x)
        return x

def load_base_model(pretrained=False):
    base_model = DarkNet(3, pretrained=pretrained)
    if not pretrained:
        print('Not loading pretrained weights...')
    if pretrained:
        print('Loading pretrained weights...')
        base_model.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/vgg11_bn-6002323d.pth'
        ))
    return base_model

def load_yolo_model(base_model, C=20, S=7, B=2):
    yolo = YOLODarknet(base_model, C=C, S=S, B=B)
    return yolo

if __name__ == '__main__':
    base_model = load_base_model(pretrained=False)
    yolo = load_yolo_model(base_model)
    print(yolo)

    x = torch.rand([1, 3, 448, 448])
    print(f"Dummy input tensor shape: {x.shape}")
    out = yolo(x)
    print(f"Output shape: {out.shape}")

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in yolo.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in yolo.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")