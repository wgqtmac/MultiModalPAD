import os

import torch
import torch.optim as optim
from torch import nn

import torch.nn.functional as F

import triplets_utils as tu


def triplet_loss(model, batch):
    model.train()
    emb = model(batch["X"].cuda())
    y = batch["y"].cuda()

    with torch.no_grad():
        triplets = tu.get_triplets(emb, y)
    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]

    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(ap_D - an_D + 1.)

    return losses.mean()


def center_loss(tgt_model, batch, src_model, src_centers, tgt_centers,
                src_kmeans, tgt_kmeans, margin=1):
    # triplets = self.triplet_selector.get_triplets(embeddings, target, embeddings_adv=embeddings_adv)
    # triplets = triplets.cuda()

    # f_N = embeddings_adv[triplets[:, 2]]

    f_N_clf = tgt_model.convnet(batch["X"].cuda()).view(batch["X"].shape[0], -1)
    f_N = tgt_model.fc(f_N_clf.detach())

    # est.predict(f_N.cpu().numpy())
    y_src = src_kmeans.predict(f_N.detach().cpu().numpy())
    # ap_distances = (emb_centers[None] - f_N[:,None]).pow(2).min(1)[0].sum(1)
    ap_distances = (src_centers[y_src] - f_N).pow(2).sum(1)
    # ap_distances = (f_C[None] - f_N[:,None]).pow(2).sum(1).sum(1)

    # an_distances = 0
    losses = ap_distances.mean()

    # y_tgt = tgt_kmeans.predict(f_N.detach().cpu().numpy())
    # ap_distances = (tgt_centers[y_tgt] - f_N).pow(2).max(1)[0]

    # losses += ap_distances.mean()*0.1

    f_P = src_model(batch["X"].cuda())
    # an_distances = (f_P - f_N).pow(2).sum(1)
    # losses -= an_distances.mean() * 0.1

    return losses