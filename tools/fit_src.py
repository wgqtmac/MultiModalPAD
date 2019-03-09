import torch.nn as nn
import losses


def fit_src(model, data_loader, opt):
    loss_sum = 0.
    for step, (images, labels) in enumerate(data_loader):
        # make images and labels variable
        images = images.cuda()
        labels = labels.squeeze_().cuda()

        # zero gradients for opt
        opt.zero_grad()

        # compute loss for critic
        loss = losses.triplet_loss(model, {"X": images, "y": labels})

        loss_sum += loss.item()

        # optimize source classifier
        loss.backward()
        opt.step()

    return {"loss": loss_sum / step}

