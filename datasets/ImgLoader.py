import os
import torch
import torch.utils.data as data
from PIL import Image
import params
import torchvision.transforms as transforms
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImgLoader(data.Dataset):
    def __init__(self, root_folder, list_file, transform=None, loader=default_loader, stage='Train'):
        self.root_folder = root_folder
        self.loader = loader
        self.transform = transform

        items = []

        fp_items = [line.rstrip('\n').split(' ') for line in open(list_file)]
        for file_name, label in fp_items:
            if os.path.isfile(os.path.join(root_folder, file_name)):
                tup = (file_name, int(label))
                items.append(tup)
        self.items = items
        print('\nStage: ' + stage)
        print('The number of samples: {}'.format(len(items)))

    def __getitem__(self, index):
        image, label = self.items[index]
        img = self.loader(os.path.join(self.root_folder, image))
        if self.transform is not None:
            img = self.transform(img)
        # img2=np.array(img)
        # print(img)
        return img, label

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    print torch.__version__
    src_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_train_list),
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomCrop(248),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                            ]))

