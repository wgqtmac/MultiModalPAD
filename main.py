"""Main script for ADDA."""
import os
import params
from model.discriminator import Discriminator
from model.ResNet80 import resnet80
from model.ResNet18 import resnet18
from model.ResNet34 import resnet34

from utils.utils import init_model, init_random_seed
import torch
import torch._utils
import torchvision.transforms as transforms
# from datasets.MultiImgLoader import ImgLoader
from datasets.TrainImgLoader import TrainImgLoader
from datasets.TestImgLoader import TestImgLoader
from datasets.Final_Score_Loader import FinalImgLoader
from datasets.ColorImgLoader import ColorImgLoader
from datasets.DepthImgLoader import DepthImgLoader
from datasets.IrImgLoader import IrImgLoader
from datasets.ImgLoader import ImgLoader
# from train_val import eval_src, eval_tgt
from model.ResNet80 import resnet80
from model.se_resnet import se_resnet18
from model.CBAM_resnet import resnet18_cbam

# from train_val import train_src, train_src_threemodal, eval_src, eval_src_score, eval_src_threemodal, eval_acc

from train_val1 import eval_src, eval_src_threemodal, eval_acc, train_feature_fusion, eval_fusionmodal, train_resnet80

import test_src as ts

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)





    train_dataset = TrainImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_train_list),
                                transforms.Compose([
                                    transforms.Resize(112),
                                    transforms.RandomAffine(10),
                                    transforms.RandomResizedCrop(size=112, scale=(0.8, 1.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                    transforms.ToTensor()
                                ]), stage='Train')

    weights = [3 if label == 1 else 1 for (image1, image2, image3, label) in train_dataset.items]
    # weights_np = np.array(weights)
    from torch.utils.data.sampler import WeightedRandomSampler

    sampler = WeightedRandomSampler(weights,
                                    num_samples=len(train_dataset.items),
                                    replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=params.batch_size,
                                             num_workers=0,
                                             sampler=sampler,
                                             pin_memory=True)

    # transforms.CenterCrop(112),
    val_dataset = TestImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_val_list),
                            transforms.Compose([
                                transforms.Resize(112),
                                transforms.CenterCrop(112),
                                transforms.ToTensor()
                            ]), stage='Test')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=params.test_batch_size,
                                             num_workers=0,
                                             pin_memory=True)

    test_dataset = FinalImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_test_list),
                                transforms.Compose([
                                    transforms.Resize(112),
                                    transforms.CenterCrop(112),
                                    transforms.ToTensor()
                                ]), stage='Test')
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=params.test_batch_size,
                                             num_workers=0,
                                             pin_memory=True)


    model = init_model(net=resnet18_cbam(),
                        restore=params.fusion_encoder_restore)


    model = train_feature_fusion(model, train_loader, val_loader)

    eval_fusionmodal(model, test_loader)

