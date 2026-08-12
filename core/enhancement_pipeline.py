"""
Low-light enhancement using Zero-DCE++ (Li, Guo, Loy - TPAMI 2021).
Embeds the official tiny (~10K parameter) architecture and loads the
authors' pretrained weights. Only runs when a frame is actually dark.
Source: https://github.com/Li-Chongyi/Zero-DCE_extension
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DARKNESS_THRESHOLD = 90  # mean V channel (0-255) below this - considered dark


class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))


class EnhanceNetNoPool(nn.Module):
    def __init__(self, scale_factor):
        super(EnhanceNetNoPool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

    def enhance(self, x, x_r):
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)
        return enhance_image

    def forward(self, x):
        x_down = x if self.scale_factor == 1 else F.interpolate(x, scale_factor=1 / self.scale_factor, mode='bilinear')
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor != 1:
            x_r = self.upsample(x_r)
        enhance_image = self.enhance(x, x_r)
        return enhance_image, x_r


class LowLightEnhancer:
    def __init__(self, weights_path: str = "models/zerodce_plusplus.pth", scale_factor: int = 12):
        self.scale_factor = scale_factor
        self.model = EnhanceNetNoPool(scale_factor)
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def needs_enhancement(self, frame_bgr) -> bool:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mean_brightness = float(np.mean(hsv[:, :, 2]))
        return mean_brightness < DARKNESS_THRESHOLD

    def enhance(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        data = frame_rgb.astype(np.float32) / 255.0
        data = torch.from_numpy(data)

        h = (data.shape[0] // self.scale_factor) * self.scale_factor
        w = (data.shape[1] // self.scale_factor) * self.scale_factor
        data = data[0:h, 0:w, :]
        data = data.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            enhanced, _ = self.model(data)

        enhanced = enhanced.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        enhanced_bgr = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return enhanced_bgr