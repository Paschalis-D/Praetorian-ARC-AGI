import torch
import torch.nn as nn

class RelationNetwork(nn.Module):
    def __init__(self):
        """
        Based on the paper 'Learning to Compare: Relation Network for Few-Shot Learning,' the Relation Network consists of
        four convolutional blocks, a feature concatenation layer, two convolutional blocks, and two fully connected layers.
        """
        super(RelationNetwork, self).__init__()
        self.block1 = nn.Sequential(
            ConvolutionalBlock(in_channels=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            ConvolutionalBlock(in_channels=64),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block3 = ConvolutionalBlock(in_channels=64)
        self.block4 = ConvolutionalBlock(in_channels=64)

        self.block5 = nn.Sequential(
            ConvolutionalBlock(in_channels=128),  # Concatenated feature maps will double the channels
            nn.MaxPool2d(kernel_size=2)
        )
        self.block6 = nn.Sequential(
            ConvolutionalBlock(in_channels=64),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(64 * 2 * 2, 8)  # Adjust the input size based on the feature map size
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x1, x2):
        # Forward pass through blocks 1-4 separately for both x1 and x2
        out1 = self.block1(x1)
        out1 = self.block2(out1)
        out1 = self.block3(out1)
        out1 = self.block4(out1)
        
        out2 = self.block1(x2)
        out2 = self.block2(out2)
        out2 = self.block3(out2)
        out2 = self.block4(out2)

        # Concatenate along the channel dimension
        out = torch.cat((out1, out2), dim=1)  # dim=1 to concatenate channels

        # Pass through blocks 5 and 6
        out = self.block5(out)
        out = self.block6(out)

        # Flatten the output for the fully connected layers
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        """
        Convolutional block with Conv2D, BatchNorm, and ReLU.
        """
        super(ConvolutionalBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    

if __name__ == "__main__":
    model = RelationNetwork()
    input1 = torch.ones(1, 1, 32, 32)
    input2 = torch.ones(1, 1, 32, 32)
    output = model(input1, input2)