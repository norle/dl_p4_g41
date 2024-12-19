import torch.nn as nn

from loss import mse_loss
from datasets import SpectrogramDataset
import torch
import torchvision.models as models
    
class SpectrVelCNNRegr(nn.Module):
    """Baseline model for regression to the velocity

    Use this to benchmark your model performance.
    """

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=37120,out_features=1024)
        self.linear2=nn.Linear(in_features=1024,out_features=256)
        self.linear3=nn.Linear(in_features=256,out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)

class YourModel(SpectrVelCNNRegr):
    """Define your model here.

    I suggest make your changes initial changes
    to the hidden layers defined in _hidden_layer below.
    This will preserve the input and output dimensionality.
    
    Eventually, you will need to update the output dimensionality
    of the input layer and the input dimensionality of your output
    layer.

    """
    def __init__(self):
        super().__init__()
        del self.conv2
        del self.conv3
        del self.conv4
        del self.linear1
        del self.linear2



    def _hidden_layer(self, x):
        """Overwrite this function"""
        pass

class MinimalModel(nn.Module):
    """Minimal model for regression to the velocity"""

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.4)
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=12,
                      kernel_size=7,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=18,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=18,
                      out_channels=24,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten=nn.Flatten()
        self.linear1=nn.Sequential(
            nn.Linear(in_features=24624, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.linear2=nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.linear3=nn.Linear(in_features=256, out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.dropout(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.flatten(x)
        x=self.dropout(x)
        x=self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)

class MinimalModel_v1(nn.Module):
    """Minimal model for regression to the velocity"""

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout2d(p=0.5)
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=3,
                      kernel_size=7,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=1,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=16,
                        out_channels=24,
                        kernel_size=3,
                        stride=1,
                        padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten=nn.Flatten()
        self.dropout_lin=nn.Dropout(p=0.5)
        self.linear1=nn.Sequential(
            nn.Linear(in_features=6960, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.linear2=nn.Sequential(
            nn.Linear(in_features=1024, out_features=50),
            nn.BatchNorm1d(50),
            nn.ReLU()
        )
        self.linear3=nn.Linear(in_features=50, out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.dropout(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.dropout_lin(x)
        x=self.linear1(x)
        x=self.dropout_lin(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)
    
class AlexNet(nn.Module):
    """AlexNet model for regression to the velocity"""

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout2d(p=0.5),

            nn.Conv2d(6, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1792, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),

        )

    def forward(self, input_data):
        x = self.features(input_data)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class ResNet(nn.Module):
    """ResNet model for regression to the velocity"""

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

    def forward(self, input_data):

        return self.resnet(input_data)
    
class ResNet_1_ch(nn.Module):
    """ResNet model for regression to the velocity"""

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.02),
            nn.Linear(512, 1)
        )

    def forward(self, input_data):

        return self.resnet(input_data)


import torch.nn.functional as F

class AttentionModel(nn.Module):
    """Attention model for regression to the velocity"""

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        self.attention = nn.Sequential(
            nn.Conv2d(256, 6, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(58368, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, input_data):
        x = F.relu(self.conv1(input_data))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        
        attention_weights = self.attention(x)
        
        x = input_data * attention_weights
        x = torch.sum(x, dim=1, keepdim=True)  # Sum all channels together
        x = torch.sum(x, dim=3)  # Sum the layer horizontally
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
class MobileNetV3(nn.Module):
    """MobileNetV3 model for regression to the velocity"""

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.mobilenet.features[0][0] = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.classifier[0].in_features, 256),
            nn.ReLU(),
            #nn.Dropout(p=0.05),
            nn.Linear(256, 1)
        )

    def forward(self, input_data):
        return self.mobilenet(input_data)

class FullyConv(nn.Module):
    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super(FullyConv, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),  # 16 x 74 x 349
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 32 x 74 x 349
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample: 32 x 37 x 174
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 x 37 x 174
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128 x 37 x 174
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample: 128 x 18 x 87
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 256 x 18 x 87
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 512 x 18 x 87
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample: 512 x 9 x 43
        )
        
        self.final_conv = nn.Conv2d(512, 1, kernel_size=1)  # 1 x 9 x 43
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: 1 x 1 x 1
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_conv(x)
        x = self.global_pool(x)
        return x  # Flatten to a single value per batch

class FullyConv_v1(nn.Module):
    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=6, stride=1, padding=3),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        )
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
        self.global_pool = nn.AdaptiveMaxPool2d((1,74))          

        self.lc = nn.Linear(74, 1)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_conv(x)
        x = self.global_pool(x)
        x = self.lc(x)
        return x  # Flatten to a single value per batch

class ShuffleNet(nn.Module):
    """ShuffleNet model for regression to the velocity"""

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        self.shufflenet = models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1)
        self.shufflenet.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.shufflenet.fc = nn.Sequential(
            nn.Linear(self.shufflenet.fc.in_features, 74),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(74, 1)
        )

    def forward(self, input_data):
        return self.shufflenet(input_data)

class AutoEncoder(nn.Module):
    """AutoEncoder model for regression to the velocity"""

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(912, 74),
            nn.ReLU()
        )
        self.out = nn.Linear(74, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.out(x)

        return x

class TwoChannel(nn.Module):
    """Baseline model for regression to the velocity

    Use this to benchmark your model performance.
    """

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=2,
                      out_channels=6,
                      kernel_size=7,
                      stride=2,
                      padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=18,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=18,
                      out_channels=54,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=12312,out_features=512)
        self.linear2=nn.Linear(in_features=512,out_features=1)

    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x= nn.ReLU()(x)
        return x

    def _output_layer(self, x):
        return self.linear2(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)



# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/n**.5
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

