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
    Triplet loss
    Takes two samples of every subject
    """

    def __init__(self, margin):
        super(PairTripletLoss, self).__init__()
        self.margin = margin

    
    def forward(self, pair1, pair2, size_average=True):
        nums, dim = pair1.shape
        losses = 0.
        cnt = 0
        for i in range(nums):
            dist_ap = (pair1[i] - pair2[i]).pow(2).sum()
            for j in range(nums):
                if i == j:
                    continue
                dist_an = (pair1[i] - pair1[j]).pow(2).sum()
                losses += F.relu(dist_ap - dist_an + self.margin)
                dist_an = (pair1[i] - pair2[j]).pow(2).sum()
                losses += F.relu(dist_ap - dist_an + self.margin)
                cnt += 1

        return losses / cnt


class OhemPairTripletLoss(nn.Module):
    """
        Ohem Triplet loss
        Takes two samples of every subject with online hard examples mining
        commit: program optimization based version 1, calculate the former number larger losses
    """

    def __init__(self, margin, number):
        super(OhemPairTripletLoss, self).__init__()
        self.margin = margin
        self.number = number


    def forward(self, pair1, pair2, size_average=True):
        nums, dim = pair1.shape
        avg_loss = 0.
        all_loss = 0.
        losses = []
        losses_np = np.zeros((self.number, ), dtype=np.float32)
        cnt = 0
        for i in range(nums):
            dist_ap = (pair1[i] - pair2[i]).pow(2).sum()
            for j in range(nums):
                if i == j:
                    continue
                dist_an1 = (pair1[i] - pair1[j]).pow(2).sum()
                loss1 = F.relu(dist_ap - dist_an1 + self.margin)
                if len(losses) == self.number:
                    alter_idx = np.argmin(losses_np)
                    if losses_np[alter_idx] < loss1.data.cpu().numpy()[0]:
                        losses[alter_idx] = loss1
                        losses_np[alter_idx] = loss1.data.cpu().numpy()[0]
                else:
                    losses.append(loss1)
                    losses_np[cnt] = loss1.data.cpu().numpy()[0]
                all_loss += loss1
                cnt += 1

                dist_an2 = (pair1[i] - pair2[j]).pow(2).sum()
                loss2 = F.relu(dist_ap - dist_an2 + self.margin)
                if len(losses) == self.number:
                    alter_idx = np.argmin(losses_np)
                    if losses_np[alter_idx] < loss2.data.cpu().numpy()[0]:
                        losses[alter_idx] = loss2
                        losses_np[alter_idx] = loss2.data.cpu().numpy()[0]
                else:
                    losses.append(loss2)
                    losses_np[cnt] = loss2.data.cpu().numpy()[0]
                all_loss += loss2
                cnt += 1

        avg_num = 0
        for i in range(self.number):
            if losses_np[i] > 0.:
                avg_loss += losses[i]
                avg_num += 1

        if avg_num > 0:
            avg_loss = avg_loss / avg_num
        else:
            avg_loss = losses[0]
        all_loss = all_loss / cnt

        return avg_loss, all_loss
