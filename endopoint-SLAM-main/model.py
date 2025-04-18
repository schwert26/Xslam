"""
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import time
class MBABlock(nn.Module):
    def __init__(self, reduction=16):
        super(MBABlock, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, 7, padding=7 // 2, bias=False)
        self.conv2 = nn.Conv2d(2, 1, 3, padding=3 // 2, bias=False)
        # 使用更高效的 Hardsigmoid 激活函数
        self.sigmoid = nn.Hardsigmoid()
        # 参数初始化
        self._initialize_weights()


    def _initialize_weights(self):
        """参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 使用 Kaiming 初始化
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 同样使用 Kaiming 初始化


    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        y = torch.cat([avg_pool, max_pool], dim=1)
        y = (self.conv1(y) + self.conv2(y)) / 2
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class BasicLayer(nn.Module):
    """
      Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
                                      nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
                                      nn.BatchNorm2d(out_channels, affine=False),
                                      nn.ReLU(inplace = True),
                                    )

    def forward(self, x):
        return self.layer(x)


class XFeatModel(nn.Module):
    """
       Implementation of architecture described in
       "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)


        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########

        self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
                                     nn.Conv2d (1, 24, 1, stride = 1, padding=0) )

        self.block1 = nn.Sequential(
                                        BasicLayer( 1,  4, stride=1),
                                        BasicLayer( 4,  8, stride=2),
                                        BasicLayer( 8,  8, stride=1),
                                        BasicLayer( 8, 24, stride=2),
                                    )

        self.block2 = nn.Sequential(
                                        BasicLayer(24, 24, stride=1),
                                        BasicLayer(24, 24, stride=1),
                                     )

        self.block3 = nn.Sequential(
                                        BasicLayer(24, 64, stride=2),
                                        BasicLayer(64, 64, stride=1),
                                        BasicLayer(64, 64, 1, padding=0),
                                     )
        #unet编码器第一个下采样模块
        self.block6 = nn.Sequential(
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                    )
        # unet编码器第二个下采样模块
        self.block7 = nn.Sequential(
                                       nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(num_features=128, affine=False),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
                                       nn.BatchNorm2d(num_features=128, affine=False),
                                       nn.ReLU(inplace=True),
                                    )
        # unet编码器第三个下采样模块
        self.block8 = nn.Sequential(
                                       nn.MaxPool2d(kernel_size=2, stride=2),
                                       nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(num_features=128, affine=False),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
                                       nn.BatchNorm2d(num_features=128, affine=False),
                                       nn.ReLU(inplace=True),
                                    )
        self.block9 = nn.Sequential(
                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
                                    #nn.ConvTranspose2d(in_channels=128, out_channels=128 , kernel_size=2, stride=2, padding=0),
                                     )

        self.block10 = nn.Sequential(
                                    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(num_features=128, affine=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128),
                                    nn.BatchNorm2d(num_features=128, affine=False),
                                    nn.ReLU(inplace=True),
                                    #nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
                                    nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
        )

        self.block11 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(num_features=64, affine=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64),
                                    nn.BatchNorm2d(num_features=64, affine=False),
                                    nn.ReLU(inplace=True),)

        self.block_fusion =  nn.Sequential(
                                        BasicLayer(64, 64, stride=1),
                                        BasicLayer(64, 64, stride=1),
                                        nn.Conv2d (64, 64, 1, padding=0)
                                     )

        self.heatmap_head = nn.Sequential(
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        nn.Conv2d (64, 1, 1),
                                        nn.Sigmoid()
                                    )


        self.keypoint_head = nn.Sequential(
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        nn.Conv2d (64, 65, 1),
                                    )
        self.mba_block=MBABlock()

    ########### ⬇️ Fine Matcher MLP ⬇️ ###########

        self.fine_matcher =  nn.Sequential(
                                            nn.Linear(128, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 128),
                                        )

    def _unfold2d(self, x, ws: int = 8):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        '''B, C, H, W = x.shape
        x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
            .reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)'''

        B, C, H, W = x.shape

        # 计算新的高度和宽度，确保它们是整数
        H_new: int = H // ws
        W_new: int = W // ws
        ws_sq: int = ws * ws

        # 使用 unfold 操作并 reshape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws)
        x = x.reshape(B, C, H_new, W_new, ws_sq)  # 确保所有参数都是 int
        x = x.permute(0, 1, 4, 2, 3).reshape(B, -1, H_new, W_new)  # 确保所有参数都是 int

        return x


    def forward(self, x):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

        """
        #dont backprop through normalization
        with torch.no_grad():
            #取rgb图片的形状是[batch_size, channels, height, width]
            #mean(dim=1, keepdim=True),dim=1指的是channels维度，取平均，设置 keepdim=True 会保留原有的维度数，只是将计算均值的维度的大小设为1。
            x = x.mean(dim=1, keepdim = True)
            #1 表示输入张量的通道数为1,对每个channels归一化，这边有几个batch_size进行几次操作
            x = self.norm(x)

        #main backbone
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))#类似与resnet的残差连接
        x3 = self.block3(x2)#(64,H/8,W/8)
        x4=  self.block7(self.block6(x3))
        x5 = self.block8(x4)
        x6 = F.interpolate(self.block9(x5), scale_factor=2.0, mode='bilinear', align_corners=False)
        x7=  torch.cat([x6,x4], dim = 1)
        x8 = F.interpolate(self.block10(x7), scale_factor=2.0, mode='bilinear', align_corners=False)
        x9 = torch.cat([x8,x3], dim = 1)
        x10= self.block11(x9)
        feats = self.block_fusion( x10 )

        #heads
        heatmap = self.heatmap_head(feats) # Reliability map
        keypoints = self.keypoint_head(self.mba_block(self._unfold2d(x))+self._unfold2d(x)) #Keypoint map logits

        return feats, keypoints, heatmap