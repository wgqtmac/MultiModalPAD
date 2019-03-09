import numpy as np
import sys
import os
import argparse

import time
import shutil
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
# import torchvision.datasets.ImageFolder as ImageFolder
import torchvision.models as models
from torch.autograd import Variable
from utils.utils import make_variable
import numpy as np
import torch.nn.functional as F
from focalloss import FocalLoss

from model.ResNet18 import resnet18

import _init_paths

from utils.utils import save_model

import params

np.set_printoptions(suppress=True)





def construct_resnet18(model, args):
    # model = resnet18()
    # model = torch.nn.DataParallel(model)

    pretrained_model = torch.load(args.pretrained_model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    model_keys = model.state_dict().keys()
    for k, v in pretrained_model.items():
        if 'fc' in k:
            continue
        if not k in model_keys:
            continue
        new_state_dict[k] = v
    state_dict = model.state_dict()
    state_dict.update(new_state_dict)
    model.load_state_dict(state_dict)

    return model


def construct_resnet34(model, args):
    # model = resnet18()
    # model = torch.nn.DataParallel(model)

    pretrained_model = torch.load(args.pretrained_model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    model_keys = model.state_dict().keys()
    for k, v in pretrained_model.items():
        if 'fc' in k:
            continue
        if not k in model_keys:
            continue
        new_state_dict[k] = v
    state_dict = model.state_dict()
    state_dict.update(new_state_dict)
    model.load_state_dict(state_dict)

    return model


# torch.cuda.set_device(3)

lr = 0.
best_prec1 = 0.

def train_feature_fusion(model, train_loader, val_loader):


    global lr
    global best_prec1

    lr = params.base_lr

    # model = construct_resnet18(model, params)
    # model = construct_resnet34(model, params)
    model.train()

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=params.base_lr,
        betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss().cuda()


    for epoch in range(params.start_epoch, params.start_epoch + params.num_epochs):
        adjust_learning_rate(optimizer, epoch, params.base_lr)

        # train for one epoch
        # train_batch(train_loader, model, criterion, optimizer, epoch)
        for step, (image1, image2, image3, label) in enumerate(train_loader):


            img1_name = image1[0][0]
            img1 = image1[1]
            img2_name = image2[0][0]
            img2 = image2[1]
            img3_name = image3[0][0]
            img3 = image3[1]
            image1 = make_variable(img1)
            image2 = make_variable(img2)
            image3 = make_variable(img3)
            label = make_variable(label.squeeze_())

            # img1_array = np.array(image1)
            # img2_array = np.array(image2)
            # img3_array = np.array(image3)
            # print(img1_name, img2_name, img3_name, img1.shape, img2.shape, img3.shape)

            feature_concat = torch.cat((image1, image2, image3), 1)

            preds = model(feature_concat)

            loss = criterion(preds, label)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("fusion Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len(train_loader),
                              loss.item()))


        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(model, val_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(model, "MultiNet-fusion-{}.pt".format(epoch + 1))


    # # save final model
    save_model(model, "MultiNet-fusion-final.pt")


    return model


def train_src(model1, model2, model3, train_loader, val_loader):


    global lr
    global best_prec1

    lr = params.base_lr

    model = construct_resnet18(model, params)
    # model = construct_resnet34(model, params)
    model.train()

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=params.base_lr,
        betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss().cuda()


    for epoch in range(params.start_epoch, params.start_epoch + params.num_epochs):
        adjust_learning_rate(optimizer, epoch, params.base_lr)

        # train for one epoch
        # train_batch(train_loader, model, criterion, optimizer, epoch)
        for step, (image1, image2, image3, label) in enumerate(train_loader):


            img1_name = image1[0][0]
            img1 = image1[1]
            img2_name = image2[0][0]
            img2 = image2[1]
            img3_name = image3[0][0]
            img3 = image3[1]
            image1 = make_variable(img1)
            image2 = make_variable(img2)
            image3 = make_variable(img3)
            label = make_variable(label.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = model1(image1)
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len(train_loader),
                              loss.item()))

        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(model, val_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(model, "MultiNet-{}.pt".format(epoch + 1))


    # # save final model
    save_model(model, "MultiNet-final.pt")


    return model

def train_src_threemodal(model1, model2, model3, train_loader1, train_loader2, train_loader3,
                         val_loader):


    global lr
    global best_prec1

    lr = params.base_lr

    # model1 = construct_resnet18(model1, params)
    # model2 = construct_resnet18(model2, params)
    # model3 = construct_resnet18(model3, params)
    model1 = construct_resnet34(model1, params)
    model2 = construct_resnet34(model2, params)
    model3 = construct_resnet34(model3, params)

    model1.train()
    model2.train()
    model3.train()

    optimizer1 = torch.optim.Adam(
        list(model1.parameters()),
        lr=params.base_lr,
        betas=(0.9, 0.99))
    optimizer2 = torch.optim.Adam(
        list(model2.parameters()),
        lr=params.base_lr,
        betas=(0.9, 0.99))
    optimizer3 = torch.optim.Adam(
        list(model3.parameters()),
        lr=params.base_lr,
        betas=(0.9, 0.99))
    # criterion = nn.CrossEntropyLoss().cuda()
    focalloss = FocalLoss(gamma=2)


    for epoch in range(params.start_epoch, params.start_epoch + params.num_epochs):
        adjust_learning_rate(optimizer1, epoch, params.base_lr)
        adjust_learning_rate(optimizer2, epoch, params.base_lr)
        adjust_learning_rate(optimizer3, epoch, params.base_lr)
        # train for one epoch
        # train_batch(train_loader, model, criterion, optimizer, epoch)
        for step, (images, labels) in enumerate(train_loader1):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer1.zero_grad()

            # compute loss for critic
            preds = model1(images)
            loss = focalloss(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer1.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Color Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len(train_loader1),
                              loss.item()))

        for step, (images, labels) in enumerate(train_loader2):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer2.zero_grad()

            # compute loss for critic
            preds = model2(images)
            loss = focalloss(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer2.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Depth Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len(train_loader2),
                              loss.item()))

        for step, (images, labels) in enumerate(train_loader3):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer3.zero_grad()

            # compute loss for critic
            preds = model3(images)
            loss = focalloss(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer3.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Ir Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len(train_loader3),
                              loss.item()))



        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_acc(model1, model2, model3, val_loader)



        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(model1, "MultiNet-color-{}.pt".format(epoch + 1))
            save_model(model2, "MultiNet-depth-{}.pt".format(epoch + 1))
            save_model(model3, "MultiNet-ir-{}.pt".format(epoch + 1))



    # # save final model
    save_model(model1, "MultiNet-color-final.pt")
    save_model(model2, "MultiNet-depth-final.pt")
    save_model(model3, "MultiNet-ir-final.pt")


    return model1, model2, model3

def eval_src(model, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    model.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.

    # for (images, labels) in data_loader:
    for (image1, image2, image3, label) in data_loader:

        img1_name = image1[0][0]
        img1 = image1[1]
        img2_name = image2[0][0]
        img2 = image2[1]
        img3_name = image3[0][0]
        img3 = image3[1]
        image1 = make_variable(img1)
        image2 = make_variable(img2)
        image3 = make_variable(img3)
        label = make_variable(label)

        feature_concat = torch.cat((image1, image2, image3), 1)

        preds = model(feature_concat)


        probability = torch.nn.functional.softmax(preds, dim=1)[:, 1].detach().tolist()


        probability1_value = np.array(probability)

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(label.data).cpu().sum()

        target_list = label.cpu().numpy()
        pred_list = pred_cls.cpu().numpy()

        for i in range(len(target_list)):
            if target_list[i] == 1 and pred_list[i] == 1:
                TP += 1
            elif target_list[i] == 0 and pred_list[i] == 0:
                TN += 1
            elif target_list[i] == 1 and pred_list[i] == 0:
                FN += 1
            elif target_list[i] == 0 and pred_list[i] == 1:
                FP += 1

    loss /= len(data_loader)
    acc = (TP + TN) / len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    print('TP:{}, TP+FN:{}, TN:{}, TN+FP:{}'.format(TP, TP + FN, TN, TN + FP))

    TP_rate = float(TP / (TP + FN))
    TN_rate = float(TN / (TN + FP))

    HTER = 1 - (TP_rate + TN_rate) / 2

    print('TP rate:{}, TN rate:{}, HTER:{}'.format(float(TP / (TP + FN)), float(TN / (TN + FP)), HTER))


def eval_src_threemodal(model1, model2, model3, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    model1.eval()
    model2.eval()
    model3.eval()


    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.

    # for (images, labels) in data_loader:
    for (image1, image2, image3, label) in data_loader:

        img1_name = image1[0][0]
        img1 = image1[1]
        img2_name = image2[0][0]
        img2 = image2[1]
        img3_name = image3[0][0]
        img3 = image3[1]
        image1 = make_variable(img1)
        image2 = make_variable(img2)
        image3 = make_variable(img3)

        preds1 = model1(image1)
        preds2 = model2(image2)
        preds3 = model3(image3)

        probability1 = torch.nn.functional.softmax(preds1, dim=1)[:, 1].detach().tolist()
        probability2 = torch.nn.functional.softmax(preds2, dim=1)[:, 1].detach().tolist()
        probability3 = torch.nn.functional.softmax(preds3, dim=1)[:, 1].detach().tolist()



        probability1_value = np.array(probability1)
        probability2_value = np.array(probability2)
        probability3_value = np.array(probability3)

        probability = (probability1_value[0] + probability1_value[0] + probability1_value[0]) / 3

        # print("{} {} {} {:.8f} {:.8f} {:.8f}".format(img1_name, img2_name, img3_name,
        #                                              probability1_value[0], probability2_value[0],
        #                                              probability3_value[0]))

        print("{} {} {} {:.8f}".format(img1_name, img2_name, img3_name,
                                       probability2_value[0]))

def eval_fusionmodal(model, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    model.eval()


    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.

    # for (images, labels) in data_loader:
    for (image1, image2, image3, label) in data_loader:
        img1_name = image1[0][0]
        img1 = image1[1]
        img2_name = image2[0][0]
        img2 = image2[1]
        img3_name = image3[0][0]
        img3 = image3[1]
        image1 = make_variable(img1)
        image2 = make_variable(img2)
        image3 = make_variable(img3)
        label = make_variable(label)


        feature_concat = torch.cat((image1, image2, image3), 1)

        preds = model(feature_concat)


        probability = torch.nn.functional.softmax(preds, dim=1)[:, 1].detach().tolist()




        probability_value = np.array(probability)


        # print("{} {} {} {:.8f} {:.8f} {:.8f}".format(img1_name, img2_name, img3_name,
        #                                              probability1_value[0], probability2_value[0],
        #                                              probability3_value[0]))

        print("{} {} {} {:.8f}".format(img1_name, img2_name, img3_name,
                                       probability_value[0]))


def eval_src_score(model, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    model.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.

    for (image1, image2, image3) in data_loader:
        img1_name = image1[0][0]
        img1 = image1[1]
        img2_name = image2[0][0]
        img2 = image2[1]
        img3_name = image3[0][0]
        img3 = image3[1]
        image1 = make_variable(img1)
        image2 = make_variable(img2)
        image3 = make_variable(img3)

        preds1 = model(image1)
        preds2 = model(image2)
        preds3 = model(image3)
        probability1 = torch.nn.functional.softmax(preds1, dim=1)[:, 0].detach().tolist()
        probability2 = torch.nn.functional.softmax(preds2, dim=1)[:, 0].detach().tolist()
        probability3 = torch.nn.functional.softmax(preds3, dim=1)[:, 0].detach().tolist()


        probability = (probability1[0] + probability2[0] + probability3[0]) / 3


    #     preds = model(images)
    #     loss += criterion(preds, labels).item()
    #
    #     # print(preds)
    #     # # prob = torch.max(F.softmax(preds), 1)[0]
    #     # print(F.softmax(preds))
    #     probability = torch.nn.functional.softmax(preds, dim=1)[:, 0].detach().tolist()
    #     # batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]
    #     print(probability)
    #     pred_cls = preds.data.max(1)[1]
    #     acc += pred_cls.eq(labels.data).cpu().sum()
    #
    #     target_list = labels.cpu().numpy()
    #     pred_list = pred_cls.cpu().numpy()
    #
    #     for i in range(len(target_list)):
    #         if target_list[i] == 1 and pred_list[i] == 1:
    #             TP += 1
    #         elif target_list[i] == 0 and pred_list[i] == 0:
    #             TN += 1
    #         elif target_list[i] == 1 and pred_list[i] == 0:
    #             FN += 1
    #         elif target_list[i] == 0 and pred_list[i] == 1:
    #             FP += 1
    #
    # loss /= len(data_loader)
    # acc = (TP + TN) / len(data_loader.dataset)
    #
    # print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    # print('TP:{}, TP+FN:{}, TN:{}, TN+FP:{}'.format(TP, TP + FN, TN, TN + FP))
    #
    # TP_rate = float(TP / (TP + FN))
    # TN_rate = float(TN / (TN + FP))
    #
    # HTER = 1 - (TP_rate + TN_rate) / 2
    #
    # print('TP rate:{}, TN rate:{}, HTER:{}'.format(float(TP / (TP + FN)), float(TN / (TN + FP)), HTER))


def eval_acc(model1, model2, model3, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    model1.eval()
    model2.eval()
    model3.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.

    # for (images, labels) in data_loader:
    for (image1, image2, image3, label) in data_loader:

        img1_name = image1[0][0]
        img1 = image1[1]
        img2_name = image2[0][0]
        img2 = image2[1]
        img3_name = image3[0][0]
        img3 = image3[1]
        image1 = make_variable(img1)
        image2 = make_variable(img2)
        image3 = make_variable(img3)
        label = make_variable(label)

        preds1 = model1(image1)
        preds2 = model2(image2)
        preds3 = model3(image3)

        probability1 = torch.nn.functional.softmax(preds1, dim=1)[:, 1].detach().tolist()
        probability2 = torch.nn.functional.softmax(preds2, dim=1)[:, 1].detach().tolist()
        probability3 = torch.nn.functional.softmax(preds3, dim=1)[:, 1].detach().tolist()

        probability = (probability1[0] + probability2[0] + probability3[0]) / 3

        probability1_value = np.array(probability1)
        probability2_value = np.array(probability2)
        probability3_value = np.array(probability3)

        pred_cls = preds1.data.max(1)[1]
        acc += pred_cls.eq(label.data).cpu().sum()

        target_list = label.cpu().numpy()
        pred_list = pred_cls.cpu().numpy()

        for i in range(len(target_list)):
            if target_list[i] == 1 and pred_list[i] == 1:
                TP += 1
            elif target_list[i] == 0 and pred_list[i] == 0:
                TN += 1
            elif target_list[i] == 1 and pred_list[i] == 0:
                FN += 1
            elif target_list[i] == 0 and pred_list[i] == 1:
                FP += 1

    loss /= len(data_loader)
    acc = (TP + TN) / len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    print('TP:{}, TP+FN:{}, TN:{}, TN+FP:{}'.format(TP, TP + FN, TN, TN + FP))

    TP_rate = float(TP / (TP + FN))
    TN_rate = float(TN / (TN + FP))

    HTER = 1 - (TP_rate + TN_rate) / 2

    print('TP rate:{}, TN rate:{}, HTER:{}'.format(float(TP / (TP + FN)), float(TN / (TN + FP)), HTER))





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global lr
    lr = base_lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# if __name__ == '__main__':
#     main()
    # extract_feature('50checkpoint.pth.tar')
