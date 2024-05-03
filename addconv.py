import torch
import torch.nn as nn

input_tensor = torch.randn(1, 3, 224, 224)

# Define convolutional layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=2, stride=2)
conv_layer2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=2, stride=2)

conv_layer3 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=4, stride=4)

# Apply convolution
# output_tensor = conv_layer(input_tensor)
# output_tensor = conv_layer2(output_tensor)

output_tensor = conv_layer3(input_tensor)

# Output tensor shape: [N, C_out, H_out, W_out]
print(output_tensor.shape)
print(output_tensor)
