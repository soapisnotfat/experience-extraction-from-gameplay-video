import torch.nn as nn
import torch.nn.functional as func


class AlexNet(nn.Module):
    def __init__(self, num_classes=30):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, (7, 7)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3)),

            nn.Conv2d(32, 128, (5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3)),

            nn.Conv2d(128, 512, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 4 * 8, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True))
        self.last_connect = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        # print("AFTER FEATURES: " + str(x.data.shape))
        x = x.view(-1, 512 * 4 * 8)
        # print("AFTER VIEW: " + str(x.data.shape))
        x = self.classifier(x)
        # print("AFTER CLASSIFIER: " + str(x.data.shape))
        x = self.last_connect(x)
        x = func.softmax(x, dim=1)
        return x
