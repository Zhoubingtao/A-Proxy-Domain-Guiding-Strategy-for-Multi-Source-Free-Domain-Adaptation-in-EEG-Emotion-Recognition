import os
import sys
import time
import torch
import scipy.io as io
# import wandb
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
import IID_losses
from model.SFDA import SFDA
from dataset.dataset_class import SFDADataset
from torch.utils.data.dataloader import DataLoader
from einops import reduce
from config.model_config import build_args, build_args_for_mutli
from utils.net_utils import set_random_seed
from utils.net_utils import init_multi_cent_psd_label, init_psd_label_shot_icml,obtain_label
from utils.net_utils import EMA_update_multi_feat_cent_with_feat_simi
from sklearn.metrics import confusion_matrix
import models

def PrPL(mem_label, dd ,all_label):

    # for i in range(len(names) - 1):
    #     io.savemat('P_mem_label'+str(i)+'.mat', {'P_mem_label': mem_label[i]})
    #     io.savemat('P_dd'+str(i)+'.mat', {'P_dd': dd[i]})
    ahp = 0.2
    all_label = all_label.to('cpu')
    all_label = all_label.numpy()
    # io.savemat('all_label.mat', {'all_label': all_label})
    de = list()
    for p in range(len(names) - 1):
        de_store = np.zeros(len(dd[p]))
        for i in range(len(dd[p])):
            DD=dd[p]
            sorted_A = sorted(DD[i, :])
            if args.method == 'BDM' :
                max1 = sorted_A[-1]
                max2 = sorted_A[-2]
                de_store[i] = max1 - max2
            else:
                min1 = sorted_A[0]
                min2 = sorted_A[1]
                de_store[i] = min2 - min1
                # max1 = sorted_A[-1]
                # max2 = sorted_A[-2]
                # de_store[i] = max1 - max2

        de.append(de_store)

    de_rest = np.zeros([len(names) - 2, len(dd[0])])

    for i in range(len(names) - 1):
        rest_id = np.arange(len(names)-1)
        rest_id = np.delete(rest_id, i)
        de_id=0
        for j in rest_id:
            de_rest[de_id, :] = de[j]
            de_id=de_id+1
        argmax_de_rest = np.argmax(de_rest,0)
        # argmax_de_rest = np.argmin(de_rest, 0)
        min_de_rest = np.zeros([len(all_label)])
        de_rest_f = np.zeros([len(all_label)])
        for label_id in range(len(all_label)):
            min_de_rest[label_id] = mem_label[rest_id[argmax_de_rest[label_id]]][label_id]
            de_rest_f[label_id] = de_rest[argmax_de_rest[label_id],label_id]
        for sample_id in range(len(all_label)):
            if de[i][sample_id] <= de_rest_f[sample_id] * ahp and de_rest_f[sample_id]!=1:
            # if de[i][sample_id] >= de_rest_f[sample_id] * ahp:
                mem_label[i][sample_id] = min_de_rest[sample_id]

        acc = np.sum(mem_label[i] == all_label) / len(all_label)
        print("交互后为 , { %f }" % (acc))

    return mem_label