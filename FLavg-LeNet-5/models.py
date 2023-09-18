import torch.nn.functional as F
import torch
import os
from torchvision import models
from torch import  nn
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def get_model(name="vgg16",pretrained=True):
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        fc_features = model.fc.in_features
        # 修改类别为10，重定义最后一层
        model.fc = nn.Linear(fc_features, 10)
    if name == "AlexNet":
        models.alexnet(pretrained=pretrained)
    if name == "resnet50":
        model = models.resnet50(pretrained=pretrained)

    if name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    if name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    if name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    if name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    if name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
    if name == "googlenet":
        model = models.googlenet(pretrained=pretrained)
    if name == "LeNet":
        model = LeNet()
    if name == "resnet":
        model = ResNet(BasicBlock, [2, 2, 2, 2])

    if name == "vgg":
        model = VGGNet()
    if name == "mlp":
        model = mlp()
    if name == "mlp1":
        model = mlp1()
    if name =="LeNet_simple":
        model=LeNet_simple()
    if name == "AlexNet":
        model = AlexNet()
    if name == "lenet_igld":
        num_classes = 10
        channel = 1
        hidden = 588
        model = lenet_igld(channel=channel, hideen=hidden, num_classes=num_classes)






    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model

class lenet_igld(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(lenet_igld, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



class CNN(nn.Module):
    def __init__(self, base=32, dense=512, num_classes=43):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, base, 3, padding=1)
        self.conv2 = nn.Conv2d(base, base, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(base, base*2, 3, padding=1)
        self.conv4 = nn.Conv2d(base*2, base*2, 3)
        self.conv5 = nn.Conv2d(base*2, base*4, 3, padding=1)
        self.conv6 = nn.Conv2d(base*4, base*4, 3)

        self.fc1 = nn.Linear(32 * 4 * 2 * 2, dense)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(dense, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = self.dropout(self.pool(x))

        x = F.relu(self.conv4(F.relu(self.conv3(x))))

        x = self.dropout(self.pool(x))

        x = F.relu(self.conv6(F.relu(self.conv5(x))))

        x = self.dropout(self.pool(x))

        x = x.view(-1, 32 * 4 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)

        x = self.fc2(x)
        return x

class mlp1(nn.Module):

    def __init__(self, input_dim=51, output_dim=2, device=None):
        super(mlp1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, output_dim)



    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)




class LeNet_simple(nn.Module):
    def __init__(self):
        super(LeNet_simple, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
        #     nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
        #     act(),
        #     nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
        #     act(),
        #     nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
        #     act(),
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(768, 100)
        # )

            nn.Conv2d(3, 32, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(32, 32, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(32, 64, kernel_size=5, padding=5 // 2, stride=1),
            act(),

        )
        self.fc = nn.Sequential(
        nn.Linear(4096, 100)
        )




    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out




class mlp(nn.Module):

	def __init__(self, input_dim=86, output_dim=2, device=None):
		super(mlp, self).__init__()
		self.fc1 = nn.Linear(input_dim, 32)
		self.fc2 = nn.Linear(32, output_dim)

		# self.linear = torch.nn.Linear(input_dim, output_dim)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)




class VGGNet(nn.Module):
    def __init__(self, num_classes=10):  # num_classes
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=False)  # 从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(512 * 7 * 7, 512),  # 512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Sequential(  # input_size=(1*28*28)
			nn.Conv2d(1, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同
			nn.ReLU(),  # input_size=(6*28*28)
			nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(6, 16, 5),
			nn.ReLU(),  # input_size=(16*10*10)
			nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
		)
		self.fc1 = nn.Sequential(
			nn.Linear(16 * 5 * 5, 120),
			nn.ReLU()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(120, 84),
			nn.ReLU()
		)
		self.fc3 = nn.Linear(84, 10)

	# 定义前向传播过程，输入为x
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		# nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
		x = x.view(x.size()[0], -1)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)

		return x





import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=43):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        # x = self.features(x)
        # h = x.view(x.shape[0], -1)
        # x = self.classifier(h)
        x = self.maxpool(self.conv1(x))
        x = self.relu(x)

        x = self.maxpool(self.conv2(x))
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.maxpool(self.conv5(x))
        x = self.relu(x)


        x = x.view(x.shape[0], -1)
        x = self.fc1(self.dropout(x))
        x = self.relu(x)

        x = self.fc2(self.dropout(x))
        x = self.relu(x)
        out = x
        x = self.fc3(x)

        return x