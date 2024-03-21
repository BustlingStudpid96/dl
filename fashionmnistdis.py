import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader


class myNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # self.mynet = nn.Sequential(
        #     nn.Linear(784, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 10),
        # )
        self.mynet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.mynet(x)

    # def evaluate(test_data, net):
    #     n_correct = 0
    #     n_total = 0
    #     with torch.no_grad():
    #         for (x, y) in test_data:
    #             outputs = net.forward(x.view(-1, 28 * 28).to(device))
    #             _, predict = torch.max(outputs.data, dim=1)
    #             n_correct += (y == predict).sum().item()
    #             n_total += y.size(0)
    #     return n_correct / n_total


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            y = y.to(device)
            outputs = net.forward(x.view(x.shape[0], 1, 28, 28).to(device))
            _, predicted = torch.max(outputs, dim=1)
            n_correct += (predicted == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total


def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    trans = torchvision.transforms.ToTensor()

    mnist_train = torchvision.datasets.FashionMNIST(
        root="../datasets/", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../datasets/", train=False, transform=trans, download=True)

    batch_size = 32
    lr = 0.002  # 开始的时候学习率太大了！
    epoch = 5
    weight_decay = 1e-6

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    mynet = myNet().to(device)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=mynet.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = 0
    state = {}
    file_name = "bestFashion.pth"
    # checkpoint = torch.load(file_name)
    # best_acc = checkpoint['accuracy']

    for i in range(epoch):

        for idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.reshape(imgs.shape[0], 1, 28, 28)
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # preds = mynet(imgs.reshape(imgs.shape[0], -1))
            preds = mynet(imgs)

            batch_loss = loss_fun(preds, labels)

            batch_loss.backward()

            optimizer.step()

        accuracy = evaluate(test_loader, mynet)
        print("epoch", i + 1, "accuracy:", accuracy)
        if accuracy > best_acc:
            best_acc = accuracy
            state['accuracy'] = best_acc
            state['state_dict'] = mynet.state_dict()
            torch.save(state, file_name)

    checkpoint = torch.load(file_name)
    best_acc = checkpoint['accuracy']
    print(f"best accuracy : {best_acc}")
    mynet.load_state_dict(checkpoint['state_dict'])
    for (n, (x, y)) in enumerate(test_loader):
        if n > 3:
            break
        true_label = get_fashion_mnist_labels([y[0].item()])
        predict = torch.argmax(mynet(x[0].view(1, 1, 28, 28).to(device)))
        predict_label = get_fashion_mnist_labels([predict.item()])
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title(f"Prediction: {predict_label}\nTrue Label: {true_label}")  # 显示预测和真实标签
    plt.show()
