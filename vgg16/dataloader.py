import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms


class CatsAndDogsData(Dataset):
    def __init__(self, root):
        self.root = root
        self.all_images = []
        images = os.listdir(root)

        for image in images:
            self.all_images.append(os.path.join(root, image))

        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        # define transforms
        self.transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ])

    def __getitem__(self, item):
        label = -1
        if self.all_images[item].__contains__("dog."):
            label = 0
        else:
            label = 1

        img = Image.open(self.all_images[item])
        img = self.transform(img)
        label = torch.tensor(label, dtype = torch.float32)

        return img, label

    def __len__(self):
        return len(self.all_images)


root = "/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/dogs-vs-cats/train"
files = os.listdir(root)

dataset = CatsAndDogsData(root)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# for data in dataloader:
#     img, label = data
#     print(label)
#
#     break