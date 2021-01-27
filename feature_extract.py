import torch
import torch.nn as N
import torchvision.models as M
import torchvision.models.resnet as R


class BlockWithLinearOutput(N.Module):
    def __init__(self, block, btype="basic"):
        super(BlockWithLinearOutput, self).__init__()

        assert btype in {"basic", "bottleneck"}

        self.btype = btype
        self.block = block

    def forward(self, x):
        residual = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        if self.btype == "bottleneck":
            out = self.block.relu(out)

            out = self.block.conv3(out)
            out = self.block.bn3(out)

        if self.block.downsample is not None:
            residual = self.block.downsample(x)

        out += residual

        return out


class ResNetFeatureExtractor(N.Module):
    """Extracts intermediate features in the given ResNet module.
    Given resnet: ResNet and feat_layer, this will perform forward inference
    until the specified layer and return the hidden features.
    Args:
        resnet (ResNet): ResNet module loaded from `torchvision.models`.
        feat_layer (str): Target feature layer specified using Caffe layer names
            proposed in the author's original code. Currently only 'res5c' is
            supported.
    """
    def __init__(self, resnet, feat_layer="res5c"):
        super(ResNetFeatureExtractor, self).__init__()

        assert feat_layer == "res5c", \
            "Current version supports only 'res5c' as the feature layer."

        self.feat_layer = feat_layer
        self.resnet = resnet
        self.layer4_stripped = N.Sequential(*[
            block for i, block in enumerate(resnet.layer4)
            if i < len(resnet.layer4) - 1
        ])
        last_block = self.resnet.layer4[-1]

        if isinstance(last_block, R.BasicBlock):
            last_block_type = "basic"
        elif isinstance(last_block, R.Bottleneck):
            last_block_type = "bottleneck"
        else:
            raise TypeError("Unexpected block type: {}".format(
                type(last_block)
            ))

        self.last_block = BlockWithLinearOutput(
            block=last_block,
            btype=last_block_type
        )

    def forward(self, x):
        """Spatial features are returned as [B x D x W x H] tensors."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.layer4_stripped(x)

        x = self.last_block(x)
        return x