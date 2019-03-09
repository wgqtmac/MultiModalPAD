import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    
    def forward(self, anchor, positive, negative, size_average=True):
        dist_ap = (anchor - positive).pow(2).sum(1)
        dist_an = (anchor - negative).pow(2).sum(1)
        losses = F.relu(dist_ap - dist_an + self.margin)
        return losses.mean() if size_average else losses.sum()


class PairTripletLoss(nn.Module):
    """
        Ohem Triplet loss
        Takes two samples of every subject with online hard examples mining
    """

    def __init__(self, margin):
        super(PairTripletLoss, self).__init__()
        self.margin = margin


    def forward(self, pair1, pair2, size_average=True):
        nums, dim = pair1.shape
        avg_loss = 0.
        all_loss = 0.
        losses = 0. 
        cnt = 0
        for i in range(nums):
            dist_ap = (pair1[i] - pair2[i]).pow(2).sum()
            sub_loss_np = 0.
            sub_loss = 0.
            for j in range(nums):
                if i == j:
                    continue
                dist_an1 = (pair1[i] - pair1[j]).pow(2).sum()
                loss1 = F.relu(dist_ap - dist_an1 + self.margin)
                if loss1.data.cpu().numpy()[0] >= sub_loss_np:
                    sub_loss_np = loss1.data.cpu().numpy()[0]
                    sub_loss = loss1
                all_loss += loss1
                cnt += 1

                dist_an2 = (pair1[i] - pair2[j]).pow(2).sum()
                loss2 = F.relu(dist_ap - dist_an2 + self.margin)
                if loss2.data.cpu().numpy()[0] >= sub_loss_np:
                    sub_loss_np = loss2.data.cpu().numpy()[0]
                    sub_loss = loss2
                all_loss += loss2
                cnt += 1
            losses += sub_loss


        sub_loss = losses / nums
        all_loss = all_loss / cnt

        return sub_loss, all_loss
