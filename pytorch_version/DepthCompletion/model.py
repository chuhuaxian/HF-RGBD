import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpBlock, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        # print('1', x.shape, concat_with.shape)
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        # print('2', up_x.shape, concat_with.shape, '\n')
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(torch.cat([up_x, concat_with], dim=1)))))


# class Decoder(nn.Module):
#     def __init__(self, num_features=1024, decoder_width=0.5):
#         super(Decoder, self).__init__()
#         features = int(num_features * decoder_width)
#
#         self.inconv = nn.Conv2d(num_features+64, features, kernel_size=1, stride=1, padding=1)  # (1024+64)*512
#
#         self.up1 = UpBlock(skip_input=features//1  + 64 + 256, output_features=features // 2)         # (512+64+256)*256
#         self.up2 = UpBlock(skip_input=features//2  + 32 + 128, output_features=features // 4)         # (256+32+128)*128
#         self.up3 = UpBlock(skip_input=features//4  + 16 +  64, output_features=features // 8)          # (128+16+64)*64
#         self.up4 = UpBlock(skip_input=features//8  +  8 +  64, output_features=features // 16)          # (64+8+64)*32
#         self.up5 = UpBlock(skip_input=features//16 +  4 +   3, output_features=features // 32)         # (32+4+3)*16
#
#         self.outconv = nn.Conv2d(features//32, 1, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, features_color, features_depth):
#
#         # 3, 64, 64, 128, 256, 1024 -> densenet121
#         # 3, 96, 96, 192, 384, 2208 -> densenet161
#         # 4, 8,  16,  32,  64,   64
#         c_in, c_block0, c_block1, c_block2, c_block3, c_block4 = \
#             features_color[0], features_color[3], features_color[4], \
#             features_color[6], features_color[8], features_color[11]
#
#         d_in, d_block0, d_block1, d_block2, d_block3, d_block4 = \
#             features_depth[0], features_depth[1], features_depth[2], \
#             features_depth[3], features_depth[4], features_depth[5]
#
#         # print('cblock4', c_block4.shape, 'dblock4', d_block4.shape, )
#         # print('cblock3', c_block3.shape, 'dblock3', d_block3.shape,)
#         # print('cblock2', c_block2.shape, 'dblock2', d_block2.shape, )
#         # print('cblock1', c_block1.shape, 'dblock1', d_block1.shape, )
#         # print('cblock0', c_block0.shape, 'dblock0', d_block0.shape, )
#         # print('cin', c_in.shape, 'dv', d_in.shape, )
#
#         x_d0 = self.inconv(torch.cat([c_block4, d_block4], dim=1))      #     1024  * 512 -> 1/32
#
#         x_d1 = self.up1(x_d0, torch.cat([c_block3, d_block3], dim=1))  # (512+256) * 256 -> 1/16
#
#         x_d2 = self.up2(x_d1, torch.cat([c_block2, d_block2], dim=1))  # (128+128) * 128 -> 1/8
#
#         x_d3 = self.up3(x_d2, torch.cat([c_block1, d_block1], dim=1))  # (64 + 64) *  64 -> 1/4
#
#         x_d4 = self.up4(x_d3, torch.cat([c_block0, d_block0], dim=1))  # (32 + 64) *  32 -> 1/2
#         x_d5 = self.up5(x_d4, torch.cat([c_in, d_in], dim=1))  # (32 + 64) *  32 -> 1/2
#         return self.outconv(x_d5)          #   32      *   1 -> 1/2


class Decoder(nn.Module):
    def __init__(self, num_features=1024, feature_base=256,  decoder_width=0.5):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.inconv = nn.Conv2d(num_features+64, features, kernel_size=1, stride=1, padding=1)  # (1024+64)*512

        self.up1 = UpBlock(skip_input=features//1  + 64 + feature_base, output_features=features // 2)         # (512+64+256)*256
        self.up2 = UpBlock(skip_input=features//2  + 32 + feature_base//2, output_features=features // 4)         # (256+32+128)*128
        self.up3 = UpBlock(skip_input=features//4  + 16 +  feature_base//4, output_features=features // 8)          # (128+16+64)*64
        self.up4 = UpBlock(skip_input=features//8  +  8 +  feature_base//4, output_features=features // 16)          # (64+8+64)*32
        self.up5 = UpBlock(skip_input=features//16 +  4 +   3, output_features=features // 32)         # (32+4+3)*16

        self.outconv = nn.Conv2d(features//32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features_color, features_depth):

        # 3, 64, 64, 128, 256, 1024 -> densenet121
        # 3, 96, 96, 192, 384, 2208 -> densenet161
        # 4, 8,  16,  32,  64,   64
        c_in, c_block0, c_block1, c_block2, c_block3, c_block4 = \
            features_color[0], features_color[3], features_color[4], \
            features_color[6], features_color[8], features_color[11]

        d_in, d_block0, d_block1, d_block2, d_block3, d_block4 = \
            features_depth[0], features_depth[1], features_depth[2], \
            features_depth[3], features_depth[4], features_depth[5]

        # print('cblock4', c_block4.shape, 'dblock4', d_block4.shape,)
        # print('cblock3', c_block3.shape, 'dblock3', d_block3.shape,)
        # print('cblock2', c_block2.shape, 'dblock2', d_block2.shape, )
        # print('cblock1', c_block1.shape, 'dblock1', d_block1.shape, )
        # print('cblock0', c_block0.shape, 'dblock0', d_block0.shape, )
        # print('cin', c_in.shape, 'dv', d_in.shape, )

        x_d0 = self.inconv(torch.cat([c_block4, d_block4], dim=1))      #     1024  * 512 -> 1/32

        x_d1 = self.up1(x_d0, torch.cat([c_block3, d_block3], dim=1))  # (512+256) * 256 -> 1/16

        x_d2 = self.up2(x_d1, torch.cat([c_block2, d_block2], dim=1))  # (128+128) * 128 -> 1/8

        x_d3 = self.up3(x_d2, torch.cat([c_block1, d_block1], dim=1))  # (64 + 64) *  64 -> 1/4

        x_d4 = self.up4(x_d3, torch.cat([c_block0, d_block0], dim=1))  # (32 + 64) *  32 -> 1/2
        x_d5 = self.up5(x_d4, torch.cat([c_in, d_in], dim=1))  # (32 + 64) *  32 -> 1/2
        return self.outconv(x_d5)


class Encoder(nn.Module):
    def __init__(self, densenet='121'):
        super(Encoder, self).__init__()       
        import torchvision.models as models
        if densenet == '161':
            self.original_model = models.densenet161(pretrained=True)
            print('Use Pretrain Densenet161 Model.')
        else:
            self.original_model = models.densenet121(pretrained=True, memory_efficient=False)
            print('Use Pretrain Densenet121 Model.')

        for k, v in self.original_model.named_parameters():
            v.requires_grad = False  # 固定参数

    def forward(self, x):
        features = [x]
        """
        Block  0 ->    3 | 1/1 
        Block  1 ->   64 | 1/2
        Block  2 ->   64 | 1/2
        Block  3 ->   64 | 1/2   **
        Block  4 ->   64 | 1/4   **
        Block  5 ->  256 | 1/4
        Block  6 ->  128 | 1/8   **
        Block  7 ->  512 | 1/8
        Block  8 ->  256 | 1/16  **
        Block  9 -> 1024 | 1/16
        Block 10 ->  512 | 1/32
        Block 11 -> 1024 | 1/32  **
        Block 12 -> 1024 | 1/32
        """
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features


class DownBlock(nn.Sequential):
    def __init__(self, input, output_features):
        super(DownBlock, self).__init__()
        self.convA = nn.Conv2d(input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)
        self.pool = nn.AvgPool2d(kernel_size=2, padding=0, ceil_mode=False)

    def forward(self, x):
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(self.pool(x)))))


class InBlock(nn.Sequential):
    def __init__(self, input, output_features):
        super(InBlock, self).__init__()
        self.convA = nn.Conv2d(input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(x))))


class Encoder_Depth(nn.Module):
    def __init__(self):
        super(Encoder_Depth, self).__init__()

        self.In = InBlock(1, 4)
        self.d0 = DownBlock(4, 8)
        self.d1 = DownBlock(8, 16)
        self.d2 = DownBlock(16, 32)
        self.d3 = DownBlock(32, 64)
        self.d4 = DownBlock(64, 64)

    def forward(self, features):
        x_in = self.In(features)  # 1*4   -> 1/1
        x_d0 = self.d0(x_in)      # 4*8   -> 1/2
        x_d1 = self.d1(x_d0)      # 8*16  -> 1/4
        x_d2 = self.d2(x_d1)      # 16*32 -> 1/8
        x_d3 = self.d3(x_d2)      # 32*64 -> 1/16
        x_d4 = self.d4(x_d3)      # 64*64 -> 1/32
        return [x_in, x_d0, x_d1, x_d2, x_d3, x_d4]


class Model(nn.Module):
    def __init__(self, pretrain_model='121'):
        super(Model, self).__init__()
        if pretrain_model == '121':
            self.encoder = Encoder(densenet='121')
            self.decoder = Decoder(num_features=1024, feature_base=256)
        else:
            self.encoder = Encoder(densenet='161')
            self.decoder = Decoder(num_features=2208, feature_base=384)

        self.encoder_depth = Encoder_Depth()

    def forward(self, x):
        color = x[:, :3, :, :]
        depth = x[:, 3:4, :, :]
        return self.decoder(self.encoder(color), self.encoder_depth(depth))

