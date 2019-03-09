"""Main script for ADDA."""
import os
import params
from model.discriminator import Discriminator
from model.ResNet80 import resnet80, ResNetClassifier
from model.ResNet18 import resnet18

from utils.utils import init_model, init_random_seed
import torch
import torch._utils
import torchvision.transforms as transforms
from datasets.MultiImgLoader import ImgLoader
from datasets.TestImgLoader import TestImgLoader
# from datasets.ImgLoader import ImgLoader
# from train_val import eval_src, eval_tgt

from train_val import train_src, eval_src, eval_src_score

import test_src as ts

os.environ["CUDA_VISIBLE_DEVICES"] = "3"



if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)




    # one model(resnet18) for 3 modal
    src_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_train_list),
                              transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.RandomCrop(248),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor()
                              ]))

    src_train_loader = torch.utils.data.DataLoader(src_dataset,
                                               batch_size=params.batch_size,
                                               num_workers=2,
                                               shuffle=True,
                                               pin_memory=True)

    src_val_dataset = TestImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_val_list),
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(248),
                                transforms.ToTensor()
                            ]), stage='Test')
    src_val_loader = torch.utils.data.DataLoader(src_val_dataset,
                                             batch_size=params.test_batch_size,
                                             num_workers=2,
                                             pin_memory=True)


    model = init_model(net=resnet18(),
                             restore=params.src_encoder_restore)




    # model = train_src(model, src_train_loader, src_val_loader)

    eval_src(model, src_val_loader)
    #eval_src_score(model, src_val_loader)

