import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class FASHION_NET(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        # print("INIT:    ", x.shape)
        x = self.conv1(x)
        # print("CONV1:   ", x.shape)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        # print("pool1:   ", x.shape)
        x = self.conv2(x)
        # print("conv2:   ", x.shape)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        # print("pool2:   ", x.shape)
        x = F.interpolate(x, size=(7,7), mode='bilinear')  # resize to the size expected by the linear unit
        # print("pool2:   ", x.shape)
        x = x.view(x.size(0), -1)
        # print("view:    ", x.shape)
        x = self.fc1(x)
        # print("fc1:     ", x.shape)
        return x


class FASHION_NET_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv25 = nn.ConvTranspose2d(8, 16, 5, bias=False, padding=2)
        self.bn45 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = F.interpolate(x, size=(7,7), mode='bilinear')  # resize to the size expected by the linear unit
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        # x = self.deconv25(x)
        # x = F.interpolate(F.leaky_relu(self.bn45(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)

        # print("FIRST:       ", x.shape)
        # x = self.conv1(x)
        # print("CONV1:       ", x.shape)
        # x = self.pool(F.leaky_relu(self.bn1(x)))
        # x = self.conv2(x)
        # print("CONV2:       ", x.shape)
        # x = self.pool(F.leaky_relu(self.bn2(x)))
        # x = F.interpolate(x, size=(7,7), mode='bilinear')  # resize to the size expected by the linear unit
        # x = x.view(x.size(0), -1)
        # print("view0:       ", x.shape)
        # x = self.fc1(x)
        # print("fc1:         ", x.shape)
        # x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        # print("view:        ", x.shape)
        # x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        # print("interpolate: ", x.shape)
        # x = self.deconv1(x)
        # print("deconv1:     ", x.shape)
        # x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        # print("interpolate2:", x.shape)
        # x = self.deconv2(x)
        # print("deconv2:     ", x.shape)
        # x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        # print("interpolate3:", x.shape)
        # x = self.deconv25(x)
        # print("deconv25:     ", x.shape)
        # x = F.interpolate(F.leaky_relu(self.bn45(x)), scale_factor=2)
        # print("interpolate35:", x.shape)
        # x = self.deconv3(x)
        # print("deconv3:     ", x.shape)
        # x = torch.sigmoid(x)
        # print("SHAPE NET:   ", x.shape)

        return x
