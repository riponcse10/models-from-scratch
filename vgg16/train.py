from model import Vgg16Model
import os
from dataloader import CatsAndDogsData
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Vgg16Model().to(device)
root = "/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/dogs-vs-cats/train"
files = os.listdir(root)

dataset = CatsAndDogsData(root)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 2

for epoch in range(epochs):
    curr_loss = 0.0
    for index, data in enumerate(dataloader):
        image, label = data
        label = label.reshape((label.shape[0], 1))
        optimizer.zero_grad()
        # label = label.float()
        print(label)
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        print(output)
        loss = criterion(output, label)

        curr_loss += loss.item()
        loss.backward()
        optimizer.step()


        print(index, curr_loss)

    print(epoch, curr_loss)

