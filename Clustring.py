import torch
import faiss
import scipy.io as io
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.cluster import SpectralClustering


def init_multi_cent_psd_label(args, model, dataloader, flag=False, flag_NRC=False, confu_mat_flag=False):
    model.eval()
    emd_feat_stack = []
    cls_out_stack = []
    gt_label_stack = []

    for data_train, data_test, data_label, data_idx in tqdm(dataloader, ncols=60, disable=True):

        data_test = data_test.cuda()
        data_label = data_label.cuda()
        if flag:
            # For G-SFDA
            embed_feat, _, cls_out = model(data_test, t=1)
        else:
            cls_out, embed_feat = model(data_test)
        emd_feat_stack.append(embed_feat)
        cls_out_stack.append(cls_out)
        gt_label_stack.append(data_label)

    all_gt_label = torch.cat(gt_label_stack, dim=0)

    all_emd_feat = torch.cat(emd_feat_stack, dim=0)
    all_emd_feat = all_emd_feat / torch.norm(all_emd_feat, p=2, dim=1, keepdim=True)
    # current VISDA-C k_seg is set to 3
    topk_num = max(all_emd_feat.shape[0] // (args.class_num * args.topk_seg), 1)

    all_cls_out = torch.cat(cls_out_stack, dim=0)
    _, all_psd_label = torch.max(all_cls_out, dim=1)
    acc = torch.sum(all_gt_label == all_psd_label) / len(all_gt_label)
    acc_list = [acc]
    # ------------------------------------------------------------#
    multi_cent_num = args.multi_cent_num
    feat_multi_cent = torch.zeros((args.class_num, multi_cent_num, args.embed_feat_dim)).cuda()
    faiss_kmeans = faiss.Kmeans(args.embed_feat_dim, multi_cent_num, niter=100, verbose=False,
                                min_points_per_centroid=1)
    # faiss_kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1, 1))
    iter_nums = 2
    for iter in range(iter_nums):
        for cls_idx in range(args.class_num):
            if iter == 0:
                # We apply TOP-K-Sampling strategy to obtain class balanced feat_cent initialization.
                feat_samp_idx = torch.topk(all_cls_out[:, cls_idx], topk_num)[1]
            else:
                # After the first iteration, we make use of the psd_label to construct feat cent.
                # feat_samp_idx = (all_psd_label == cls_idx)
                feat_samp_idx = torch.topk(feat_dist[:, cls_idx], topk_num)[1]

            feat_cls_sample = all_emd_feat[feat_samp_idx, :].cpu().numpy()
            faiss_kmeans.train(feat_cls_sample)
            feat_multi_cent[cls_idx, :] = torch.from_numpy(faiss_kmeans.centroids).cuda()

        feat_dist = torch.einsum("cmk, nk -> ncm", feat_multi_cent, all_emd_feat)  # [N,C,M]
        feat_dist, _ = torch.max(feat_dist, dim=2)  # [N, C]
        feat_dist = torch.softmax(feat_dist, dim=1)  # [N, C]

        _, all_psd_label = torch.max(feat_dist, dim=1)
        acc = torch.sum(all_psd_label == all_gt_label) / len(all_gt_label)
        acc_list.append(acc)
    #
    # feat_dist_save = feat_dist.cpu()
    # feat_dist_save = feat_dist_save.detach().numpy()
    # io.savemat('feat_dist_save.mat', {'feat_dist_save': feat_dist_save})
    #
    # all_psd_label_save = all_psd_label.cpu()
    # all_psd_label_save = all_psd_label_save.detach().numpy()
    # io.savemat('all_psd_label_save.mat', {'all_psd_label_save': all_psd_label_save})
    #
    # all_gt_label_save = all_gt_label.cpu()
    # all_gt_label_save = all_gt_label_save.detach().numpy()
    # io.savemat('all_gt_label_save.mat', {'all_gt_label_save': all_gt_label_save})

    log = "acc:" + " --> ".join("{:.3f}".format(acc) for acc in acc_list)
    psd_confu_mat = confusion_matrix(all_gt_label.cpu(), all_psd_label.cpu())
    psd_acc_list = psd_confu_mat.diagonal() / psd_confu_mat.sum(axis=1) * 100
    psd_acc = psd_acc_list.mean()
    psd_acc_str = "{:.2f}        ".format(psd_acc) + " ".join(["{:.2f}".format(i) for i in psd_acc_list])

    if args.test:
        print(log)
    else:
        print(log)

    if args.dataset == "VisDA":
        print(psd_acc_str)

    if flag or flag_NRC:
        # For G-SFDA or NRC
        return feat_multi_cent, all_psd_label, all_emd_feat, all_cls_out
    else:
        # For SHOT and SHOT++
        if confu_mat_flag:
            return feat_multi_cent, all_psd_label, psd_confu_mat
        else:
            return feat_dist, feat_multi_cent, all_psd_label, all_gt_label