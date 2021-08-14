from gcommand_dataset import GCommandLoader
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import numpy as np
import torch.utils.data as data
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms

dataset = GCommandLoader('train')
trainLoader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, pin_memory=True)
validDataset = GCommandLoader('valid')
validLoader = torch.utils.data.DataLoader(validDataset, batch_size=128, shuffle=False, pin_memory=True)
transforms = transforms.Compose([transforms.Normalize((0.0307,), (0.3081,))])
testDataset = GCommandLoader('test', transform=transforms)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False, pin_memory=True)
map = dataset.class_to_idx


class BestNet(nn.Module):
    # setting up the network
    def __init__(self):
        super(BestNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 8, 1, 1).cuda()
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1).cuda()
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1).cuda()
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1).cuda()
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1).cuda()
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1).cuda()
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1).cuda()

        self.bnc1 = nn.BatchNorm2d(64).cuda()
        self.bnc2 = nn.BatchNorm2d(64).cuda()
        self.bnc3 = nn.BatchNorm2d(128).cuda()
        self.bnc4 = nn.BatchNorm2d(128).cuda()
        self.bnc5 = nn.BatchNorm2d(256).cuda()
        self.bnc6 = nn.BatchNorm2d(256).cuda()
        self.bnc7 = nn.BatchNorm2d(256).cuda()

        self.pool = nn.MaxPool2d(2, 2).cuda()

        self.fc1 = nn.Linear(512, 256).cuda()
        self.fc2 = nn.Linear(256, 128).cuda()
        self.fc3 = nn.Linear(128, 30).cuda()

        self.bn1 = nn.BatchNorm1d(256).cuda()
        self.bn2 = nn.BatchNorm1d(128).cuda()
        self.bn3 = nn.BatchNorm1d(30).cuda()

    # forward propagation
    def forward(self, x):
        x = F.relu(self.pool(self.bnc1(self.conv1(x))))
        x = F.relu(self.bnc2(self.conv2(x)))
        x = F.relu(self.bnc3(self.conv3(x)))
        x = F.relu(self.pool(self.bnc4(self.conv4(x))))
        x = F.relu(self.pool(self.bnc5(self.conv5(x))))
        x = F.relu(self.pool(self.bnc6(self.conv6(x))))
        x = F.relu(self.pool(self.bnc7(self.conv7(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        return x


# setting up the model
model = BestNet()
model = model.cuda()
model.to('cuda:0')
criterion = CrossEntropyLoss().cuda()  # negative log likelihood
optimizer = optim.Adam(model.parameters(), lr=0.000095)
validLossMin = np.inf
epochs = 12
train_losses, valid_losses, train_acc, valid_acc = [], [], [], []
# train the model for 10 epochs
for epoch in range(epochs):
    runningLoss = 0
    validationLoss = 0
    hits = 0
    correct = 0
    val_hits = 0
    model.train()
    for audio, label in trainLoader:
        audio = audio.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        output = model(audio.cuda())
        loss = criterion(output, label.cuda())
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        hits += pred.eq(label.cuda().view_as(pred)).sum()
        runningLoss += loss.item() * audio.size(0)
    runningLoss /= len(trainLoader.sampler)
    # print(
    #     '\nepoch Number {:d}: Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch, runningLoss,
    #                                                                                                hits,
    #                                                                                                len(
    #                                                                                                    trainLoader.sampler),
    #                                                                                                100. * hits / len(
    #                                                                                                    trainLoader.sampler)))

    model.eval()
    with torch.no_grad():
        for validAudio, validLabel in validLoader:
            optimizer.zero_grad()
            output = model(validAudio.cuda())
            loss = criterion(output, validLabel.cuda())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(validLabel.cuda().view_as(pred)).sum()
            validationLoss += loss.item() * validAudio.size(0)  # sum up batch loss
        validationLoss /= len(validLoader.sampler)
        # print(
        #     '\nepoch Number {:d}: Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        #                                                                                                     validationLoss,
        #                                                                                                     correct,
        #                                                                                                     len(
        #                                                                                                         validLoader.sampler),
        #                                                                                                     100. * correct / len(
        #                                                                                                         validLoader.sampler)))


# get the path
def stripString(givenString):
    while True:
        try:
            index = givenString.index('\\')
            givenString = givenString[index + 1:]
        except:
            return givenString


# test on test set
def test(model, test_loader, map):
    map = {v: k for k, v in map.items()}
    modelPredictions = []
    model.eval()
    currentCounter, lineCounter = 0, 0
    length = len(test_loader.sampler)
    outputFile = open("test_y", "w")
    with torch.no_grad():
        for data, labels in test_loader:
            output = model(data.cuda())
            predictions = output.max(1, keepdim=True)[1]
            for prediction in predictions.data:
                path = stripString(test_loader.sampler.data_source.spects[currentCounter][0])
                number = int(path.split(".wav")[0])
                modelPredictions.append([number, path, (map[prediction.item()])])
            currentCounter += 1
        modelPredictions.sort()
    for finalPrediction in modelPredictions:
        finalPredictionString = str(finalPrediction[1]) + "," + str(finalPrediction[2])
        if lineCounter != length - 1:
            outputFile.write(finalPredictionString + "\n")
            lineCounter += 1
        else:
            outputFile.write(finalPredictionString)
    outputFile.close()


test(model, testLoader, map)
