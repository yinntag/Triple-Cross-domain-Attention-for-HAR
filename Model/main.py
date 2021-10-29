import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
import os
from trip_att import TripletAttention
import matplotlib.pyplot as plt
import pylab as pl

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# use_cuda=torch.cuda.is_available()


# train_x = np.load('/mnt/experiment/tangyin/TransCNN_HAR/Multi_CT/model/LOSO/pamap2/train_x09.npy')
# train_x = np.load('/mnt/experiment/tangyin/Time Window/pamap2/data/train_x445.npy')
train_x = np.load('/mnt/experiment/tangyin/LegoNet-master/data_uci/pamap2/train_x.npy')
shape = train_x.shape
train_x = torch.from_numpy(np.reshape(train_x.astype(float), [shape[0], 1, shape[1], shape[2]]))
train_x = train_x.type(torch.FloatTensor).cuda()
print(train_x.shape)
print("-" * 100)

# train_y = (np.load('/mnt/experiment/tangyin/TransCNN_HAR/Multi_CT/model/LOSO/pamap2/train_y09.npy'))
# train_y = np.load('/mnt/experiment/tangyin/Time Window/pamap2/data/train_y445.npy')
train_y = np.load('/mnt/experiment/tangyin/LegoNet-master/data_uci/pamap2/train_y.npy')
# train_y = np.asarray(pd.get_dummies(train_y))
# train_y = np.asarray(pd.get_dummies(train_y))
train_y = torch.from_numpy(train_y)
train_y = train_y.type(torch.FloatTensor).cuda()
print(train_y.shape)
print("-" * 100)


# test_x = np.load('/mnt/experiment/tangyin/TransCNN_HAR/Multi_CT/model/LOSO/pamap2/test_x09.npy')
# test_x = np.load('/mnt/experiment/tangyin/Time Window/pamap2/data/test_x445.npy')
test_x = np.load('/mnt/experiment/tangyin/LegoNet-master/data_uci/pamap2/test_x.npy')



test_x = torch.from_numpy(np.reshape(test_x.astype(float), [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]))
test_x = test_x.type(torch.FloatTensor).cuda()

# test_y = np.load('/mnt/experiment/tangyin/TransCNN_HAR/Multi_CT/model/LOSO/pamap2/test_y09.npy')
# test_y = np.load('/mnt/experiment/tangyin/Time Window/pamap2/data/test_y445.npy')
test_y = np.load('/mnt/experiment/tangyin/LegoNet-master/data_uci/pamap2/test_y.npy')
# test_y = np.asarray(pd.get_dummies(test_y))
test_y = torch.from_numpy(test_y.astype(np.float32))
test_y = test_y.type(torch.FloatTensor).cuda()
print(test_y.shape)

# print(train_x.shape, train_y.shape)
torch_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=512, shuffle=True, num_workers=0)
torch_dataset = Data.TensorDataset(test_x, test_y)
test_loader = Data.DataLoader(dataset=torch_dataset, batch_size=512, shuffle=True, num_workers=0)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        conv1 = nn.Conv2d(1, 64, (6, 1), (3, 1), padding=(1, 0))
        att1 = TripletAttention()

        conv2 = nn.Conv2d(64, 128, (6, 1), (3, 1), padding=(1, 0))
        att2 = TripletAttention()

        conv3 = nn.Conv2d(128, 256, (6, 1), (3, 1), padding=(1, 0))
        att3 = TripletAttention()

        self.conv_module = nn.Sequential(
            conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            att1,

            conv2,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            att2,

            conv3,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            att3
        )
        self.classifier = nn.Sequential(
            nn.Linear(51200, 12),
            # nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.conv_module(x)
        x = torch.flatten(x, 1)
        # print(x.shape, 'x')
        x = self.classifier(x)
        return x


class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.att1 = TripletAttention()

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
        )
        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.att2 = TripletAttention()

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
        )
        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.att3 = TripletAttention()

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
        )
        self.fc = nn.Sequential(
            nn.Linear(51200, 12)
        )

    def forward(self, x):
        # print(x.shape)
        out1 = self.Block1(x)
        out1 = self.att1(out1)
        y1 = self.shortcut1(x)
        out = y1 + out1

        out2 = self.Block2(out)
        out2 = self.att2(out2)
        y2 = self.shortcut2(out)
        out = y2 + out2

        out3 = self.Block3(out)
        out3 = self.att3(out3)
        y3 = self.shortcut3(out)
        out = y3 + out3

        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda()
        return out

model = resnet().cuda()
print(model)

# learning_rate = 0.005
learning_rate = 5e-4


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


def plot_confusion(comfusion,class_data):
    plt.figure(figsize=(15, 12))
    # plt.rcParams['figure.dpi'] = 1000
    plt.rcParams['font.family'] = ['Times New Roman']
    classes = class_data
    print(len(classes))
    plt.imshow(comfusion, interpolation='nearest', cmap=plt.cm.Greens)  # 按照像素显示出矩阵
    plt.title('Confusion Matrix for PAMAP2 dataset', fontsize=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=315)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=12)
    thresh = comfusion.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    print('11111')
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (comfusion.size, 2))
    print('11111')
    for i, j in iters:
        plt.text(j, i, format(comfusion[i, j]))  # 显示对应的数字

    plt.ylabel('Real Label', fontsize=12)
    plt.xlabel('Prediction Label', fontsize=12)

    plt.tight_layout()
    # plt.savefig('/home/tangyin/桌面/emnist/LegoNet-master/data_uci/pamap2/pamap2_confusion3.png', dpi=350)
    plt.show()


def train(epoch):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))
    adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    model.train()
    # print(model)
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        labels = labels.long().cpu()
        # labels = np.argmax(labels, axis=1)
        # print(images.size())
        # print(labels.size())
        # print("labels = ", labels)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output.cpu(), labels.cpu())
        loss_list.append(loss.data.item())
        batch_list.append(i + 1)

        if i == 1:
            print('Current Learning_rate: ', optimizer.param_groups[0]['lr'])
            print('Training: Epoch %d,  Loss: %f' % (epoch, loss.data.item()))

        loss.backward(retain_graph=True)
        optimizer.step()


acc = 0
acc_best = 0


def test(test_acc, epoch):
    global acc, acc_best
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            labels = labels.long().cpu()
            # labels = np.argmax(labels, axis=1)
            output = model(images)
            avg_loss += criterion(output.cpu(), labels).cuda().sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.cuda().data.view_as(pred)).cuda().sum()
    avg_loss /= len(test_x)
    acc = float(total_correct) / len(test_x)
    if acc_best < acc:
        acc_best = acc
    print('Testing: Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    test_acc.append(acc)
    # np.save("/mnt/experiment/tangyin/Triplet_Attention/pamap2/loso1.npy", test_acc)
    print('-' * 50)


def train_and_test(epoch):
    train(epoch)
    test(test_acc, epoch)

test_acc = []


def main():
    epoch = 200
    for e in range(0, epoch):
        train_and_test(e)


if __name__ == '__main__':
    main()