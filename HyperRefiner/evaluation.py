# -*- coding：UTF-8 -*-
'''
@Project : DIP-HyperKite-main
@File ：evaluation.py
@Author : Zerbo
@Date : 2023/6/6 23:08
'''
import os
import argparse
import json
import torch
import numpy as np
from torch.nn.functional import threshold, unfold
from dataloaders.HSI_datasets import *
from utils.logger import Logger
import torch.utils.data as data
from utils.helpers import initialize_weights, initialize_weights_new, to_variable, make_patches
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models.models import MODELS
from utils.metrics import *
import shutil
import torchvision
from torch.distributions.uniform import Uniform
import sys

from scipy.io import savemat
import torch.nn.functional as F
from utils.vgg_perceptual_loss import VGGPerceptualLoss, VGG19
from utils.spatial_loss import Spatial_Loss
from models.VAEtest import vaeLoss
from models.HyperRefiner import HyperAE

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate(band_set):
    d_lambda = 0.0
    d_s = 0.0
    qnr = 0.0
    model.eval()
    pred_dic = {}
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            MS_image, PAN_image = data

            MS_image = MS_image.float().cuda()
            PAN_image = PAN_image.float().cuda()

            # Taking model output
            out = model(PAN_image, MS_image)
            outputs = out["pred"]

            # Scalling
            outputs[outputs < 0] = 0.0
            outputs[outputs > 1.0] = 1.0

            #MS_image = torch.round(MS_image.detach()*config[config["train_dataset"]]["max_value"])
            #PAN_image = torch.round(PAN_image.detach()*config[config["train_dataset"]]["max_value"])

            ### Computing performance metrics ###
            # D_lambda for an image
            D_l = 0.0
            for j in range(iters):
                D_l += D_lambda(outputs, MS_image, band_set[j])
            D_l = D_l/iters

            # D_s for an image
            ds = D_s(outputs, MS_image, PAN_image, config[config["train_dataset"]]["factor"])

            # QNR
            Qnr = QNR(D_l, ds)

            print(D_l)
            print(ds)
            print(Qnr)
            qnr += Qnr
            d_lambda += D_l
            d_s += ds

            outputs = torch.round(outputs * config[config["train_dataset"]]["max_value"])
            pred_dic.update({"beijing_" + str(i) + "_pred": torch.squeeze(outputs).permute(1, 2, 0).cpu().numpy()})
    print(d_lambda)
    print(d_s)
    print(qnr)
    d_lambda /= len(test_loader)
    d_s /= len(test_loader)
    qnr /= len(test_loader)


    # Return Outputs
    metrics = {
        "d_lambda": float(d_lambda),
        "d_s":      float(d_s),
        "qnr":      float(qnr),
        }
    return pred_dic, metrics

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == "__main__":

    __dataset__ = {
        "pavia_dataset":    pavia_dataset, "botswana_dataset": botswana_dataset,
        "chikusei_dataset": chikusei_dataset, "botswana4_dataset": botswana4_dataset,
        "beijing_dataset": beijing_dataset
        }

    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument(
            '-c', '--config', default='configs/config_eval.json', type=str, help='Path to the config file'
            )
    parser.add_argument(
            '-r', '--resume', default=None, type=str, help='Path to the '
                                                           '.pth model checkpoint to resume training'
            )
    parser.add_argument(
            '-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)'
            )
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    # LOADING THE CONFIG FILE
    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True

    # NUMBER OF GPUs
    num_gpus = torch.cuda.device_count()

    # MODEL
    #model = MODELS[config["model"]](config)
    model = HyperAE(config)
    print(f'\n{model}\n')

    # SENDING MODEL TO DEVICE
    if num_gpus > 1:
        print("Training with multiple GPUs ({})".format(num_gpus))
        model = nn.DataParallel(model).cuda()
    else:
        print("Single Cuda Node is avaiable")
        model.cuda()

    # DATA LOADERS
    print("Evaluating with dataset => {}".format(config["train_dataset"]))
    test_loader = data.DataLoader(
            __dataset__[config["train_dataset"]](
                    config, is_train=False, want_DHP_MS_HR=config["is_DHP_MS"], ), batch_size=1,
            num_workers=config["num_workers"], shuffle=False, pin_memory=False, )

    # RESUME, loading the well-trained model parameters
    if config['model'] != "HyAE":
        print("Loading from other FCN and copying weights to continue....")
        checkpoint = torch.load(config[config["train_dataset"]][config["model"]])
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("Loading from HyperRefiner model.pth and copying weights to continue....")
        checkpoint = torch.load(config[config["train_dataset"]]["HyAE"])
        model.load_state_dict(checkpoint, strict=False)
        #if use a sepratedlly trained ae model.pth, load to
        #ae_checkpoint = torch.load(config[config["train_dataset"]]["AE"])
        #model.AE.load_state_dict(checkpoint, strict=False)

    # SETTING UP TENSORBOARD and COPY JSON FILE TO SAVE DIRECTORY
    PATH = "./" + config["experim_name"] + "/" + config["train_dataset"] + \
           "/" + "_" + str(
            config["N_modules"]
            )
    ensure_dir(PATH + "/")
    writer = SummaryWriter(log_dir=PATH)
    shutil.copy2(args.config, PATH)


    print("\n Evaluating")
    #for hsi, evaluation for a set of bands that are selected randomly
    band_set=[]
    N_bands = config[config["train_dataset"]]["spectral_bands"]
    iters = 1 #num of ev
    selected_num = 145
    assert selected_num <= N_bands
    for i in range(iters):
        arr = np.arange(N_bands)
        #random.seed(0)
        rng = np.random.default_rng(i)
        rng.shuffle(arr)
        #np.random.shuffle(arr)
        band_list = arr[0: selected_num]
        band_list.sort()
        band_set.append(band_list)

    #print(band_set)
    pred_dic, metrics = evaluate(band_set)
    # Saving best performance metrics
    with open(PATH + "/" + "eval_metrics.json", "w+") as outfile:
        json.dump(metrics, outfile)

    # Saving prediction
    savemat(PATH + "/" + "evaluation.mat", pred_dic)

