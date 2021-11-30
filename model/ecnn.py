import numpy as np
import torch
import torch.nn as nn


class ECNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=7, pretrained=False):
        super(ECNN, self).__init__()
        if pretrained:
            warnings.warn("Pretrained model is not available")
        self.relu = nn.ReLU(inplace=True)
        self.max = nn.MaxPool2d((2,2))
        self.conv1d = nn.Conv1d(in_channel, 16, kernel_size=97, stride=1, padding=48)
        
        self.dilated_conv1_1 = nn.Conv2d(1, 8, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.dilated_conv1_2 = nn.Conv2d(1, 8, (3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
        
        self.dilated_conv2_1 = nn.Conv2d(16, 8, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.dilated_conv2_2 = nn.Conv2d(16, 8, (3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
        
        self.dilated_conv3_1 = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.dilated_conv3_2 = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 2), dilation=(1, 2))
        
        self.dilated_conv4_1 = nn.Conv2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.dilated_conv4_2 = nn.Conv2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 2), dilation=(1, 2))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 1 * 256, 100),
            nn.Dropout(p=0.5),
            self.relu,
            nn.Linear(100, out_channel)
        )
    
    
    def forward(self, x):
        # (batch_size,1,4096)
        x = self.conv1d(x)  # (batch_size,16,4096)
        x = self.relu(x)
        x = x.unsqueeze(1)  # (batch_size,1,16,4096)
        
        conv1_1 = self.dilated_conv1_1(x)
        conv1_2 = self.dilated_conv1_2(x)
        x = torch.cat([conv1_1, conv1_2], 1)
        x = self.max(self.relu(x))# (batch_size,16,8,2048)

        conv2_1 = self.dilated_conv2_1(x)
        conv2_2 = self.dilated_conv2_2(x)
        x = torch.cat([conv2_1, conv2_2], 1)
        x = self.max(self.relu(x))# (batch_size,16,4,1024)
        
        conv3_1 = self.dilated_conv3_1(x)
        conv3_2 = self.dilated_conv3_2(x)
        x = torch.cat([conv3_1, conv3_2], 1)
        x = self.max(self.relu(x))# (batch_size,32,2,512)
        
        conv4_1 = self.dilated_conv4_1(x)
        conv4_2 = self.dilated_conv4_2(x)
        x = torch.cat([conv4_1, conv4_2], 1)
        x = self.max(self.relu(x))# (batch_size,32,1,256)
        
        x = x.flatten(1)
        x = self.fc_layers(x)
        
        return x


class ECNN_fft(nn.Module):
    def __init__(self, in_channel=1, out_channel=7, pretrained=False):
        super(ECNN_fft, self).__init__()
        if pretrained:
            warnings.warn("Pretrained model is not available")
        self.relu = nn.ReLU(inplace=True)
        self.max = nn.MaxPool2d((2, 2))
        self.conv1d = nn.Conv1d(in_channel, 16, kernel_size=97, stride=1, padding=48)
        
        self.dilated_conv1_1 = nn.Conv2d(1, 8, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.dilated_conv1_2 = nn.Conv2d(1, 8, (3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
        
        self.dilated_conv2_1 = nn.Conv2d(16, 8, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.dilated_conv2_2 = nn.Conv2d(16, 8, (3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
        
        self.dilated_conv3_1 = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.dilated_conv3_2 = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 2), dilation=(1, 2))
        
        self.dilated_conv4_1 = nn.Conv2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.dilated_conv4_2 = nn.Conv2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 2), dilation=(1, 2))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 1 * 128, 100),
            nn.Dropout(p=0.5),
            self.relu,
            nn.Linear(100, out_channel)
        )
    
    def forward(self, x):
        # (batch_size,1,4096)
        x = self.conv1d(x)  # (batch_size,16,4096)
        x = self.relu(x)
        x = x.unsqueeze(1)  # (batch_size,1,16,4096)
        
        conv1_1 = self.dilated_conv1_1(x)
        conv1_2 = self.dilated_conv1_2(x)
        x = torch.cat([conv1_1, conv1_2], 1)
        x = self.max(self.relu(x))  # (batch_size,16,8,2048)
        
        conv2_1 = self.dilated_conv2_1(x)
        conv2_2 = self.dilated_conv2_2(x)
        x = torch.cat([conv2_1, conv2_2], 1)
        x = self.max(self.relu(x))  # (batch_size,16,4,1024)
        
        conv3_1 = self.dilated_conv3_1(x)
        conv3_2 = self.dilated_conv3_2(x)
        x = torch.cat([conv3_1, conv3_2], 1)
        x = self.max(self.relu(x))  # (batch_size,32,2,512)
        
        conv4_1 = self.dilated_conv4_1(x)
        conv4_2 = self.dilated_conv4_2(x)
        x = torch.cat([conv4_1, conv4_2], 1)
        x = self.max(self.relu(x))  # (batch_size,32,1,256)
        
        x = x.flatten(1)
        x = self.fc_layers(x)
        
        return x

if __name__ == '__main__':#测试数据格式
    test = torch.nn.Parameter(torch.reshape(torch.rand(2048*2), (2, 1, 2048)))
    model = ECNN_fft()
    probs = model(test)
    print(probs.shape, '\n' , probs)