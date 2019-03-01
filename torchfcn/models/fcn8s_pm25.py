import torch.nn as nn
import torch

class FCN8sPM25(nn.Module):

    def __init__(self, feature_dim=87):
        super(FCN8sPM25, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(feature_dim, 16, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)

        # conv2
        self.conv2_1 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)

        # conv3
        self.conv3_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)

        # conv4
        self.conv4_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)

        # conv5
        self.conv5_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)

        # fc6
        self.fc6 = nn.Conv2d(128, 1024, 7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        # self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.relu7 = nn.ReLU(inplace=True)
        # self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(1024, 1, 1)
        self.score_conv3 = nn.Conv2d(64, 1, 1)
        self.score_conv4 = nn.Conv2d(128, 1, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.uniform_(-0.1, 0.1)
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        conv3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        conv4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.relu6(self.fc6(h))
        h = self.relu7(self.fc7(h))

        score_fr = self.score_fr(h)
        score_conv4c = self.score_conv4(conv4)
        score_conv3c = self.score_conv3(conv3)

        h = score_conv4c + score_conv3c  + score_fr

        return h
    
    
    
class FCN8sPM25_2conv(nn.Module):

    def __init__(self, feature_dim=87):
        super(FCN8sPM25_2conv, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(feature_dim, 16, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)


        self.fc = nn.Conv2d(16, 1, 1)
        # self.relu_fc = nn.ReLU(inplace=True)
        # self.drop7 = nn.Dropout2d()

        # self.score_fr = nn.Conv2d(16, 1, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            pass
            # 随机初始化(方案1)
            # if isinstance(m, nn.Conv2d):
            #     # m.weight.data.zero_()
            #     m.weight.data.uniform_(-0.1, 0.1)
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            # 随机初始化(方案2)
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))


    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.fc(h)

        return h
    
    
class FCN8sPM25_1conv(nn.Module):

    def __init__(self, feature_dim=87):
        super(FCN8sPM25_1conv, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(feature_dim, 1, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            pass

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        return h
    
    
class FCN8sPM25_1conv_size_1_1(nn.Module):

    def __init__(self, feature_dim=87):
        super(FCN8sPM25_1conv_size_1_1, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(feature_dim, 256, 1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(256, 1, 1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            pass

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        return h
    
    
class FCN8sPM25_inception1(nn.Module):

    def __init__(self, feature_dim=87):
        super(FCN8sPM25_inception1, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(feature_dim, 16, 1)
        self.conv1_2 = nn.Conv2d(feature_dim, 16, 3, padding=1)
        self.conv1_3 = nn.Conv2d(feature_dim, 16, 5, padding=2)

        self.relu1_1 = nn.Sigmoid()
        # self.relu1_1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(48, 1, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            pass

    def forward(self, x):
        x0 = self.conv1_1(x)
        x1 = self.conv1_2(x)
        x2 = self.conv1_3(x)
        out = torch.cat((x0, x1, x2), 1)
        x3 = self.relu1_1(out)
        x4 = self.conv2(x3)
        
        return x4



if __name__ == "__main__":
    model = FCN8sPM25_inception1()
    model.eval()
    image = torch.randn(1, 87, 55, 55)

    print(model)
    print("input shape:", image.shape)
    print("output shape:", model(image).shape)
    print("output:", model(image))
    
    # print(model)
    # print(model.conv1_1.weight)
