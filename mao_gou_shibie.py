import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

train_data_dir = './datas/cat-dog/train'
test_data_dir = './datas/cat-dog/val'

train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
    # transforms.RandomRotation(degrees=(-10, 10)),  #随机旋转，-10到10度之间随机选
    # transforms.RandomHorizontalFlip(p=0.5),  #随机水平翻转 选择一个概率概率
    # transforms.RandomVerticalFlip(p=0.5),  #随机垂直翻转
    # transforms.RandomPerspective(distortion_scale=0.6, p=1.0), # 随机视角
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  #随机选择的高斯模糊模糊图像
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    transforms.Normalize(  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
])

test_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_data_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)


class MyNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(MyNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # torch.Size([4, 16, 53, 53])
            nn.Flatten(),  # torch.Size([4, 44944])
            nn.Linear(16 * 53 * 53, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2)
        )

    def forward(self, x):
        x = self.net(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
mynet = MyNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mynet.parameters(), lr=1e-3)


def train_fun(train_loader, mynet, loss_fn, optimizer, device):
    for i, (X, y) in enumerate(train_loader):

        mynet.train()
        X, y = X.to(device), y.to(device)

        pred = mynet(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"i : {i} , loss : {loss}")


def test_fun(test_loader, mynet, loss_fn, device):
    mynet.eval()
    test_loss, n_total, n_correct, accuracy = 0, 0, 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            pred = mynet(X)
            test_loss += loss_fn(pred, y)
            n_correct += (torch.max(pred, dim=1)[1] == y).sum().item()
            n_total += y.size(0)
    accuracy = n_correct / n_total
    print(f"accuracy : {accuracy}")


epochs = 20
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_fun(train_loader, mynet, loss_fn, optimizer, device)
    test_fun(test_loader, mynet, loss_fn, device)
print("Done!")


def get_cat_dog_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['cat', 'dog']
    return text_labels[labels]


mean = [0.485, 0.456, 0.406],
std = [0.229, 0.224, 0.225]
# 展示图片
for n, (X, y) in enumerate(test_loader):
    X, y = X.to(device), y.to(device)

    predict = mynet(X)

    predicted_idx = torch.argmax(predict, 1)

    # 获取第一张图片的预测标签和真实标签
    predict_label = get_cat_dog_labels(predicted_idx[0].item())
    true_label = get_cat_dog_labels(y[0].item())

    # 因为X是一个batch的tensor，我们只取第一张图片
    image = X[0]  # 取出第一张图片
    # 转换为numpy数组并调整形状以匹配imshow的期望（彩色图片）
    image_np = image.permute(1, 2, 0).cpu().numpy() * std + mean
    # 显示图片
    plt.imshow(image_np)
    plt.title(f"Batch {n + 1}, Prediction: {predict_label}\nTrue Label: {true_label}")
    plt.axis('off')
    plt.show()

    if n >= 4:
        break
