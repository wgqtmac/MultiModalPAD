import torch.nn as nn
import torch
import losses
import triplets_utils as tu
from sklearn.cluster import KMeans


def fit_disc(src_model, tgt_model, disc,
             src_loader, tgt_loader,
             opt_tgt, opt_disc,
             epochs=200,
             verbose=1):
    tgt_model.train()
    disc.train()

    # setup criterion and opt
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(epochs):
        # zip source and target data pair

        data_zip = enumerate(zip(src_loader, tgt_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = images_src.cuda()
            images_tgt = images_tgt.cuda()

            # zero gradients for opt
            opt_disc.zero_grad()

            # extract and concat features
            feat_src = src_model.get_embedding(images_src)
            feat_tgt = tgt_model.get_embedding(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = disc(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.ones(feat_src.size(0)).long()
            label_tgt = torch.zeros(feat_tgt.size(0)).long()
            label_concat = torch.cat((label_src, label_tgt), 0).cuda()

            # compute loss for disc
            loss_disc = criterion(pred_concat, label_concat)
            loss_disc.backward()

            # optimize disc
            opt_disc.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for opt
            opt_disc.zero_grad()
            opt_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_model.get_embedding(images_tgt)

            # predict on discriminator
            pred_tgt = disc(feat_tgt)

            # prepare fake labels
            label_tgt = torch.ones(feat_tgt.size(0)).long().cuda()

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            opt_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if verbose and ((step + 1) % 20 == 0):
                print("Epoch [{}/{}] - step[{}]"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              epochs,
                              step+1,
                              loss_disc.item(),
                              loss_tgt.item(),
                              acc.item()))


def fit_center(src_model, tgt_model, src_loader, tgt_loader,
               opt_tgt, epochs=30):
    """Center loss."""
    ####################
    # 1. setup network #
    ####################
    n_classes = 2
    # set train state for Dropout and BN layers
    src_model.train()
    tgt_model.train()

    src_embeddings, _ = tu.extract_embeddings(src_model, src_loader)

    src_kmeans = KMeans(n_clusters=n_classes)
    src_kmeans.fit(src_embeddings)

    # src_centers = torch.FloatTensor(src_kmeans.means_).cuda()
    src_centers = torch.FloatTensor(src_kmeans.cluster_centers_).cuda()

    ####################
    # 2. train network #
    ####################

    for epoch in range(epochs):
        for step, (images, labels) in enumerate(tgt_loader):
            # make images and labels variable
            images = images.cuda()
            labels = labels.squeeze_().cuda()

            # zero gradients for opt
            opt_tgt.zero_grad()

            # compute loss for critic
            loss = losses.center_loss(tgt_model, {"X": images, "y": labels}, src_model,
                                      src_centers, None, src_kmeans,
                                      None)
            # optimize source classifier
            loss.backward()
            opt_tgt.step()
