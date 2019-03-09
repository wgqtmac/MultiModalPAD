import os
import torch
import torch.utils.data as data
from PIL import Image
import params
import torchvision.transforms as transforms
import numpy as np


def RGB_loader(path):
    return Image.open(path).convert('RGB')

def YCbCr_loader(path):
    return Image.open(path).convert('YCbCr')

def Gray_loader(path):
    return Image.open(path).convert('L')

def default_loader(path):
    return Image.open(path).convert('RGB')


# class TrainImgLoader(data.Dataset):
#     def __init__(self, root_folder, list_file, transform=None, loader1=RGB_loader, loader2=Gray_loader, stage='Train'):
#         self.root_folder = root_folder
#         self.loader1 = loader1
#         self.loader2 = loader2
#         self.transform = transform
#
#         items = []
#
#         fp_items = [line.rstrip('\n').split(' ') for line in open(list_file)]
#         for file_color, file_depth, file_ir, label in fp_items:
#             if os.path.isfile(os.path.join(root_folder, file_color)) and \
#             os.path.isfile(os.path.join(root_folder, file_depth)) and \
#             os.path.isfile(os.path.join(root_folder, file_ir)):
#                 tup = (file_color, file_depth, file_ir, int(label))
#                 items.append(tup)
#         self.items = items
#         print('\nStage: ' + stage)
#         print('The number of samples: {}'.format(len(fp_items)))
#
#     def __getitem__(self, index):
#         image1, image2, image3, label = self.items[index]
#         img1 = self.loader1(os.path.join(self.root_folder, image1))
#         img2 = self.loader2(os.path.join(self.root_folder, image2))
#         img3 = self.loader2(os.path.join(self.root_folder, image3))
#         if self.transform is not None:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
#             img3 = self.transform(img3)
#
#         # img1 = np.array(img1)
#         # img2 = np.array(img2)
#         # img3 = np.array(img3)
#         return (image1, img1), (image2, img2), (image3, img3), label
#
#     def __len__(self):
#         return len(self.items)

class TrainImgLoader(data.Dataset):
    def __init__(self, root_folder, list_file, transform=None, loader1=RGB_loader, loader2=Gray_loader ,stage='Train'):
        self.root_folder = root_folder
        self.loader1 = loader1
        self.loader2 = loader2

        self.transform = transform

        items = []

        fp_items = [line.rstrip('\n').split(' ') for line in open(list_file)]
        for file_color, file_depth, file_ir, label in fp_items:
            if os.path.isfile(os.path.join(root_folder, file_color)) and \
            os.path.isfile(os.path.join(root_folder, file_depth)) and \
            os.path.isfile(os.path.join(root_folder, file_ir)):
                tup = (file_color, file_depth, file_ir, int(label))
                items.append(tup)
        self.items = items
        print('\nStage: ' + stage)
        print('The number of samples: {}'.format(len(fp_items)))

    def __getitem__(self, index):
        image1, image2, image3, label = self.items[index]
        img1 = self.loader1(os.path.join(self.root_folder, image1))
        img2 = self.loader2(os.path.join(self.root_folder, image2))
        img3 = self.loader2(os.path.join(self.root_folder, image3))

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)



        # img1 = np.array(img1)
        # img2 = np.array(img2)
        # img3 = np.array(img3)
        return (image1, img1), (image2, img2), (image3, img3), label

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    print torch.__version__
    src_dataset = TrainImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_train_list),
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomCrop(248),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                            ]))

