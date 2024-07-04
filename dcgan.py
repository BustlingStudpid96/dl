import torch
import torchvision
import torch.nn as nn
import numpy as np

embedding_dim = 64
batch_size = 64
num_classes = 10
num_epochs = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"正在使用{device}")
# 这表示潜在向量（latent vector）的大小为96。
# 这里的大小96意味着这个向量有96个维度。
nz = 100
# 这表示图像中的通道数为3。对于CIFAR-10数据集来说，图像是RGB格式的，因此有3个通道：红色、绿色和蓝色。
nc = 3  # Number of channels in the images (CIFAR-10 is RGB)
# 这表示生成器中第一层卷积转置层（ConvTranspose2d）的滤波器（或称为卷积核）的数量为64。
ngf = 64  # Number of generator filters in first conv layer
# 这表示判别器中第一层卷积层（Conv2d）的滤波器数量为64。
ndf = 64  # Number of discriminator filters in first conv layer


class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        # inchanels = nz + embedding_dim
        self.model1 = nn.Sequential(
            nn.Linear(164, ngf * 8 * 1 * 1),  # 512*1*1?
            nn.BatchNorm1d(ngf * 8 * 1 * 1)
        )
        self.model2 = nn.Sequential(
            # # 第一个转置卷积层 z:[64,132,1,1]
            # nn.ConvTranspose2d(in_channels=ngf * 16,  # 256+32=132
            #                    out_channels=ngf * 8,
            #                    kernel_size=4, stride=1, padding=0),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(inplace=True),
            # 第二个转置卷积层 [64,512,4,4]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # 第三个转置卷积层 [64,256,8,8]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # 第四个转置卷积层 [64,128,16,16]
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # 第五个转置卷积层 [64,64,32,32]
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1),
            # # The size of tensor a (64) must match the size of tensor b (32) at non-singleton dimension 3
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # 使用1*1卷积改变通道数
            # nn.Conv2d(64, 3, kernel_size=1),
            # [64,3,64,64]
            nn.Tanh()
        )

    def forward(self, z, labels):
        labels = self.embedding(labels)
        z = torch.cat([z, labels], dim=-1)  # [64,164]
        z = self.model1(z)  # [64,16384]
        z = z.reshape(z.size(0), ngf * 8, 1, 1)  # [64,1024,4,4]
        z = self.model2(z)

        # # # 裁剪中心32*32作为生成图片？
        # start_h = (z.size(2) - 32) // 2
        # start_w = (z.size(3) - 32) // 2
        # cropped_z = z[:, :, start_h:start_h + 32, start_w:start_w + 32]
        # # print(cropped_z.shape)
        # return cropped_z
        return z


class Discriminator(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.model = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            # 第二个卷积层
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            # nn.GELU(),
            # 第三个卷积层
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            # 第四个卷积层
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            # 将特征图展平
            nn.Flatten(),
            # 接收展平后的特征图和类别标签的嵌入
            # 512是卷积层之后的特征图的即通道数
            # 经过前面的卷积层和池化层之后，特征图的大小被缩小到了2*2。
            nn.Linear(512 * 2 * 2 + embedding_dim, 1),  # (2080,1)
            # nn.Sigmoid()  # 输出一个介于0和1之间的值
        )

    def forward(self, images, labels):
        # print(images.shape,labels.shape)  torch.Size([64, 3, 32, 32]) torch.Size([64])

        embedding_labels = self.embedding(labels)  # shape[64,32]

        # 取出除了最后两个层之外的所有层（即不包括Linear,sigmoid）
        images = self.model[:-1](images)
        # 拼接images和labels
        images = torch.cat([images, embedding_labels], dim=1)
        # print(images.shape)  # [64,2080] 之后在进入后两层处理
        images = self.model[-1:](images)
        # print(images.shape) # [64,1]
        return images


# Training
dataset_path = '../../1_17/pythonProject/datasets/'
dataset = torchvision.datasets.CelebA(root=dataset_path, download=True, transform=torchvision.transforms.Compose(
    [
        # torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
        #                                  std=[0.2023, 0.1994, 0.2010]),
        torchvision.transforms.Normalize(mean=0.5, std=0.5),
    ]
)
                                       )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=0.0002, weight_decay=0.0001)
d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.0002, weight_decay=0.0001)
# g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
# d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

# loss_fn = nn.BCELoss()
loss_fn = nn.MSELoss()
labels_one = torch.ones(batch_size, 1, device=device)
labels_zero = torch.zeros(batch_size, 1, device=device)

for epoch in range(num_epochs):
    for i, (gt_images, labels) in enumerate(dataloader):
        gt_images = gt_images.to(device)
        labels = labels.to(device)

        if i % 3 == 0:
            z = torch.randn(batch_size, nz, device=device)  # [64,256]
            pred_images = generator(z, labels)
            g_optimizer.zero_grad()
            # recons_loss = torch.abs(pred_images - gt_images).mean()
            recons_loss = nn.functional.l1_loss(gt_images, pred_images)
            # 生成器希望判别器将自己生成的图片判断为真的 11111
            g_loss = recons_loss * 0.05 + loss_fn(discriminator(pred_images, labels), labels_one)
            g_loss.backward()
            g_optimizer.step()

        # if i % 3 == 0:
        d_optimizer.zero_grad()
        # 判别器有两个loss，真图片要判断为真11111，生成的图片要判断为假00000
        real_loss = loss_fn(discriminator(gt_images, labels), labels_one)
        fake_loss = loss_fn(discriminator(pred_images.detach(), labels), labels_zero)
        d_loss = (real_loss + fake_loss)
        d_loss.backward()
        d_optimizer.step()

        if i % 50 == 0:
            print(
                f"epoch:{epoch} , step:{len(dataloader) * epoch + i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")

        if i % 400 == 0:
            image = pred_images[:16].data
            torchvision.utils.save_image(image, f"./gimgs/cifa/image_{len(dataloader) * epoch + i}.png", nrow=4)
# generator = Generator()
# labels = torch.randint(1, 10, size=[batch_size, ])
# print(labels.shape)  # 64
# embedding = nn.Embedding(num_classes, embedding_dim)
# labels = embedding(labels)
# print(labels.shape)  # torch.Size([64, 32])  64个样本的类别都分别被映射到32维的向量中了
# z = torch.randn(batch_size, nz)  # [64, 96]
# z = torch.cat([z, labels], dim=-1)  # [64, 128]
# z = z.reshape(batch_size, nz + embedding_dim, 1, 1)
# z = generator(labels)
# print("generator 后 z :" , z.shape)  # torch.Size([64, 3, 32, 32])
# discriminator = Discriminator()
# discriminator(z, labels)
