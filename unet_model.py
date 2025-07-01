#%%

"""
Author: Harshith Bachimanchi
Date: 2023-02-03
Description: UNet model
"""

import torch.nn as nn
import torch, torchvision


class DoubleConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super().__init__()
        self.relu = relu
        self.double_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.double_convolution_block_wihout_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.relu:
            x = self.double_convolution_block(x)
        else:
            x = self.double_convolution_block_wihout_relu(x)
        return x


class UNetEncoder(nn.Module):
    """
    Encoder for UNet: Halves the spatial dimensions and doubles the number of channels.
    """

    def __init__(
        self,
        input_shape=(1, 3, 572, 572),
        conv_layer_dimensions=(64, 128, 256, 512, 1024),
    ):
        super().__init__()
        self.input_shape = input_shape
        self.encoder_block1 = DoubleConvolutionBlock(
            input_shape[1], conv_layer_dimensions[0]
        )
        self.encoder_blocks = nn.ModuleList(
            [
                DoubleConvolutionBlock(
                    conv_layer_dimensions[i],
                    conv_layer_dimensions[i + 1],
                )
                for i in range(len(conv_layer_dimensions) - 1)
            ]
        )
        self.pool = nn.MaxPool2d(2, padding=0)

    def forward(self, x):
        feature_maps = []
        x = self.encoder_block1(x)
        feature_maps.append(x)
        x = self.pool(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            feature_maps.append(x)
            x = self.pool(x)
        return feature_maps


class UNetDecoder(nn.Module):
    """
    Decoder for UNet: Doubles the spatial dimensions and halves the number of channels.
    """

    def __init__(self, conv_layer_dimensions=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.conv_layer_dimensions = conv_layer_dimensions
        # upconvolution layers
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    conv_layer_dimensions[i],
                    conv_layer_dimensions[i + 1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                )
                for i in range(len(conv_layer_dimensions) - 1)
            ]
        )
        # double convolution layers after upconvolution
        self.decoder_blocks = nn.ModuleList(
            [
                DoubleConvolutionBlock(
                    conv_layer_dimensions[i],
                    conv_layer_dimensions[i + 1],
                )
                for i in range(len(conv_layer_dimensions) - 2)
            ]
        )
        # final double convolution layer should not have relu
        self.last_decoder_block = DoubleConvolutionBlock(
            conv_layer_dimensions[-2], conv_layer_dimensions[-1], relu=False
        )

    def forward(self, x, feature_maps):
        for i in range(len(self.conv_layer_dimensions) - 1):
            x = self.upconvs[i](x)
            features_maps_crops = self.crop(feature_maps[i], x)
            x = torch.cat([x, features_maps_crops], dim=1)
            if i == len(self.conv_layer_dimensions) - 2:  # when i = 3 i.e, 128
                x = self.last_decoder_block(x)
            else:
                x = self.decoder_blocks[i](x)
        return x

    def crop(self, feature_maps, x):
        _, _, H, W = x.shape
        cropped_feature_maps = torchvision.transforms.CenterCrop((H, W))(feature_maps)
        cropped_feature_maps = feature_maps
        return cropped_feature_maps


class UNet(nn.Module):
    def __init__(
        self,
        input_shape=(1, 1, 128, 128),
        conv_layer_dimensions=(16, 32, 64, 128, 256),
        number_of_output_channels=1,
    ):
        super().__init__()
        self.encoder = UNetEncoder(
            input_shape=input_shape, conv_layer_dimensions=conv_layer_dimensions
        )
        self.decoder = UNetDecoder(
            conv_layer_dimensions=conv_layer_dimensions[::-1]
        )  # reverse the list
        self.final_conv = nn.Conv2d(
            conv_layer_dimensions[0], number_of_output_channels, 3, padding=1
        )
        # Batch_norm?

    def forward(self, x):
        feature_maps = self.encoder(x)
        x = self.decoder(feature_maps[-1], feature_maps[::-1][1:])
        x = self.final_conv(x)
        return x


class IntermediateLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IntermediateLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))



class UNetZ_split(nn.Module):
    def __init__(
        self,
        input_shape=(1, 1, 128, 128),
        conv_layer_dimensions=(16, 32, 64, 128, 256),
        number_of_output_channels_detect=2,
        number_of_output_channels_z=2,
        ):
        super().__init__()
        self.encoder = UNetEncoder(input_shape=input_shape, conv_layer_dimensions=conv_layer_dimensions)
        self.decoder = UNetDecoder(conv_layer_dimensions=conv_layer_dimensions[::-1])
        
        # Intermediate layer before final layers for each branch
        intermediate_channels = conv_layer_dimensions[0]  # or any other number of channels you choose
        self.intermediate_for_pos = IntermediateLayer(intermediate_channels, intermediate_channels)
        self.intermediate_for_z = IntermediateLayer(intermediate_channels, intermediate_channels)
        

        # Final layers for each branch
        self.final_conv_for_pos = nn.Conv2d(intermediate_channels, number_of_output_channels_detect, 3, padding=1)  # Particle position prediction
        self.final_conv_for_z = nn.Conv2d(intermediate_channels, number_of_output_channels_z, 3, padding=1)    # Z-position prediction
        
    def forward(self, x):
        feature_maps = self.encoder(x)
        x = self.decoder(feature_maps[-1], feature_maps[::-1][1:])
        
        # Pass through intermediate layers
        x_pos = self.intermediate_for_pos(x)
        x_z = self.intermediate_for_z(x)
        
        # Final predictions
        pos_pred = self.final_conv_for_pos(x_pos)
        z_pred = self.final_conv_for_z(x_z)
        
        # Concatenate or stack predictions along the channel dimension if necessary
        # Note: Depending on your application, you might return them separately or combined
        return torch.cat([pos_pred, z_pred], dim=1)

# %%
# model = UNet(
#     input_shape=(1, 1, 512, 512),
#     conv_layer_dimensions=(64,128,256,512,1024),
#     number_of_output_channels=1,
# )

# input_test = torch.randn(1, 1, 512, 512)
# output_test = model(input_test)
# print(output_test.shape)

# # save onnx model
# torch.onnx.export(
#     model,
#     input_test,
#     "unet.onnx",
# )

# from torchinfo import summary

# model = UNet(
#     input_shape=(1, 1, 256, 256),
#     conv_layer_dimensions=(8, 16, 32),
#     number_of_output_channels=1,
# )

# summary(model, input_size=(1, 1, 256, 256))


# %%
