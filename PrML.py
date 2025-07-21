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

def PrML(mem_label, dd ,all_label, all_output_save):

    # for i in range(len(names) - 1):
    #     io.savemat('P_mem_label'+str(i)+'.mat', {'P_mem_label': mem_label[i]})
    #     io.savemat('P_dd'+str(i)+'.mat', {'P_dd': dd[i]})

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


        de.append(de_store)
    de_np = np.array(de)

    max_id = np.argmax(de_np , axis=0)

    mem_label_np = np.transpose(np.array(mem_label))

    mem_label_best = mem_label_np[np.arange(len(max_id)),max_id]


    dd_np = np.array(all_output_save)
    de_best = dd_np[max_id, np.arange(len(max_id)), :]
    best=np.argmax(de_best, 1)
    acc_best = np.mean(best == all_label)


    print(f'ACC：{acc_best:.2f}')
    return de_best