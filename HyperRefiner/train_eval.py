# -*- coding：UTF-8 -*-
'''
@Project : hypervae
@File ：train_eval.py
@Author : Zerbo
@Date : 2023/6/7 19:03
'''
# -*- coding：UTF-8 -*-
'''
@Project : hypervae
@File ：train_hyae.py
@Author : Zerbo
@Date : 2022/11/10 15:10
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
import sys
from scipy.io import savemat
import torch.nn.functional as F
from utils.spatial_loss import Spatial_Loss
from models.HyperRefiner import HyperAE

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


# Training epoch.
def train_ae(epoch):
    train_loss = 0.0
    model.train()
    optimizer_ae.zero_grad()
    optimizer_rest.zero_grad()
    for i, data in enumerate(train_loader, 0):
        # Reading data.
        if config["is_DHP_MS"]:
            MS_image, PAN_image, MS_dhp = data
        else:
            MS_image, PAN_image= data

        # Taking model outputs ...
        MS_image = Variable(MS_image.float().cuda())  # 8*145*30*30  BChw
        PAN_image = Variable(PAN_image.float().cuda())  # 8*120*120  BHW

        if config["is_DHP_MS"]:
            MS_dhp = Variable(MS_dhp.float().cuda())
            out = model(PAN_image, MS_image, MS_dhp)
        else:
            out = model(PAN_image, MS_image, None)
        coarse_hsi = out['coarse_hsi']
        outputs = out['pred']
        de_lr_hsi = out['de_lr_hsi']

        ######### Computing loss #########
        # sr loss

        # AE_loss
        if config['ae_loss']:
            # max_ms = torch.amax(MS_image, dim=(2, 3)).unsqueeze(2).unsqueeze(3).expand_as(MS_image).cuda()
            ae_loss = config['ae_loss_F'] * criterion(de_lr_hsi, to_variable(MS_image))

        torch.autograd.backward(ae_loss)

        if i % config["trainer"]["iter_size"] == 0 or i == len(train_loader) - 1:
            optimizer_ae.step()
            optimizer_ae.zero_grad()
            optimizer_rest.step()
            optimizer_rest.zero_grad()  # optimizer_ae.module.step()  # optimizer_ae.zero_grad()  # optimizer_rest.module.step()  # optimizer_rest.zero_grad()

    writer.add_scalar('Loss/train', ae_loss, epoch)


# Testing epoch.
def test_ae(epoch):
    test_loss = aux_test_loss = 0.0
    cc = cc_aux = 0.0
    sam = sam_aux = 0.0
    rmse = rmse_aux = 0.0
    ergas = ergas_aux = 0.0
    psnr = psnr_aux = 0.0
    val_outputs = {}
    model.eval()
    pred_dic = {}
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            if config["is_DHP_MS"]:
                image_dict, MS_image, PAN_image, reference, MS_dhp = data
            else:
                MS_image, PAN_image = data

            # Inputs and references...
            MS_image = MS_image.float().cuda()
            PAN_image = PAN_image.float().cuda()


            if config["is_DHP_MS"]:
                MS_dhp = MS_dhp.float().cuda()
                out = model(PAN_image, MS_image, MS_dhp)
            else:
                out = model(PAN_image, MS_image, None)
            # Taking model output
            de_lr_hsi = out['de_lr_hsi']

            # Computing validation loss
            loss = criterion(de_lr_hsi, MS_image)
            test_loss += loss.item()


            # Scalling
            outputs = de_lr_hsi
            outputs[outputs < 0] = 0.0
            outputs[outputs > 1.0] = 1.0
            outputs = torch.round(outputs * config[config["train_dataset"]]["max_value"])

            #pred_dic.update({image_dict["imgs"][0].split("/")[-1][:-4] + "_pred": torch.squeeze(outputs).permute(1, 2, 0).cpu().numpy()})
            reference = torch.round(MS_image.detach() * config[config["train_dataset"]]["max_value"])

            ### Computing performance metrics ###
            # Cross-correlation
            cc += cross_correlation(outputs, reference)

            # SAM
            sam += SAM(outputs, reference)

            # RMSE
            rmse += RMSE(outputs / torch.max(reference), reference / torch.max(reference))

            # ERGAS
            beta = torch.tensor(config[config["train_dataset"]]["HR_size"] / config[config["train_dataset"]]["LR_size"]).cuda()
            ergas += ERGAS(outputs, reference, beta)

            # PSNR
            psnr += PSNR(outputs, reference)


    # Taking average of performance metrics over test set
    cc /= len(test_loader)
    sam /= len(test_loader)
    rmse /= len(test_loader)
    ergas /= len(test_loader)
    psnr /= len(test_loader)

    # Writing test results to tensorboard
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Test_Metrics/CC', cc, epoch)
    writer.add_scalar('Test_Metrics/SAM', sam, epoch)
    writer.add_scalar('Test_Metrics/RMSE', rmse, epoch)
    writer.add_scalar('Test_Metrics/ERGAS', ergas, epoch)
    writer.add_scalar('Test_Metrics/PSNR', psnr, epoch)



    # Images to tensorboard
    # Normalizing the images
    outputs = outputs / torch.max(reference)

    reference = reference / torch.max(reference)

    pred = torch.unsqueeze(outputs.view(-1, outputs.shape[-2], outputs.shape[-1]), 1)

    ref = torch.unsqueeze(reference.view(-1, reference.shape[-2], reference.shape[-1]), 1)
    imgs = torch.zeros(5 * pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])

    imgs = torchvision.utils.make_grid(imgs, nrow=5)
    writer.add_image('Images', imgs, epoch)

    # Return Outputs
    metrics_pred = {"loss": float(test_loss), "cc": float(cc), "sam": float(sam), "rmse": float(rmse), "ergas": float(ergas), "psnr": float(psnr)}


    return metrics_pred


if __name__ == "__main__":
    __dataset__ = {"pavia_dataset": pavia_dataset, "botswana_dataset": botswana_dataset, "chikusei_dataset": chikusei_dataset,
                   "botswana4_dataset": botswana4_dataset, "botswana2_dataset": botswana2_dataset, "botswana5_dataset": botswana5_dataset, "botswana75_dataset": botswana75_dataset,
                   "beijing_dataset": beijing_dataset}
    device_ids = [0,]
    # Parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_train_hyae_beiijng.json', type=str, help='Path to the config file')
    parser.add_argument('-r', '--resume', default="G:\\ZB\\workfile\\work\\hypervae\\Experiments\\hypae_wp\\botswana_dataset\\sr\\best_model.pth", type=str, help='Path to the ae.pth model checkpoint to resume training')
    #"G:\\ZB\\workfile\\work\\hypervae\\Experiments\\hypae_wp\\chikusei_dataset\\sr\\best_model1.pth"
    #"G:\\ZB\\workfile\\work\\hypervae\\Experiments\\hypae_wp\\pavia_dataset\\sr\\best_model1.pth"
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    # Loading the config file
    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True

    # Set seeds.
    torch.manual_seed(7)

    # Setting number of GPUS available for training.
    num_gpus = torch.cuda.device_count()

    # Selecting the model.
    # model = HyperAE(config)
    model = HyperAE(config)
    print(f'\n{model}\n')

    # Setting up training and testing dataloaderes.
    print("Training with dataset => {}".format(config["train_dataset"]))
    train_loader = data.DataLoader(__dataset__[config["train_dataset"]](config, is_train=True, want_DHP_MS_HR=config["is_DHP_MS"], ),
                                   batch_size=config["train_batch_size"], num_workers=config["num_workers"], shuffle=True, pin_memory=False, drop_last=False)

    test_loader = data.DataLoader(__dataset__[config["train_dataset"]](config, is_train=False, want_DHP_MS_HR=config["is_DHP_MS"], ),
                                  batch_size=config["val_batch_size"], num_workers=config["num_workers"], shuffle=True, pin_memory=False, )

    # Initialization of hyperparameters.
    start_epoch = 1
    total_epochs = config["trainer"]["total_epochs"]

    # Setting up optimizer for ae and the rest
    ae_params = list(map(id, model.AE.parameters()))
    rest_params = filter(lambda p: id(p) not in ae_params, model.parameters())

    if config["optimizer"] == "SGD":
        optimizer_ae = optim.SGD(model.AE.parameters(), lr=config["AE"]["optimizer"]["args"]["lr"],
                                 weight_decay=config["AE"]["optimizer"]["args"]["weight_decay"])
        optimizer_rest = optim.SGD(rest_params, lr=config["sr"]["optimizer"]["args"]["lr"], weight_decay=config["sr"]["optimizer"]["args"]["weight_decay"])
    elif config["optimizer"] == "ADAM":
        optimizer_ae = optim.Adam(model.AE.parameters(), lr=config["AE"]["optimizer"]["args"]["lr"],
                                  weight_decay=config["AE"]["optimizer"]["args"]["weight_decay"])
        optimizer_rest = optim.Adam(rest_params, lr=config["sr"]["optimizer"]["args"]["lr"], weight_decay=config["sr"]["optimizer"]["args"]["weight_decay"])
    else:
        exit("Undefined optimizer type")

    # Learning rate sheduler.
    scheduler_ae = optim.lr_scheduler.StepLR(optimizer_ae, step_size=config["AE"]["optimizer"]["step_size"], gamma=config["AE"]["optimizer"]["gamma"])
    scheduler_rest = optim.lr_scheduler.StepLR(optimizer_rest, step_size=config["sr"]["optimizer"]["step_size"], gamma=config["sr"]["optimizer"]["gamma"])
    # Sending model to GPU  device.
    if num_gpus > 1:
        print("Training with multiple GPUs ({})".format(num_gpus))
        model = nn.DataParallel(model).cuda()
        optimizer_ae = nn.DataParallel(optimizer_ae, device_ids=device_ids)
        optimizer_rest = nn.DataParallel(optimizer_rest, device_ids=device_ids)
    else:
        print("Single Cuda Node is avaiable")
        model.cuda()

    # Resume...
    if config[config["train_dataset"]]["pre_trained_ae"] == "None" or config[config["train_dataset"]]["pre_trained_sr"] == "None":
        initialize_weights(model)
    if config[config["train_dataset"]]["pre_trained_ae"] != "None":
        print("Loading from existing Ae_FCN and copying weights to continue....")
        checkpoint = torch.load(config[config["train_dataset"]]["pre_trained_ae"])
        model.load_state_dict(checkpoint, strict=False)
    if config[config["train_dataset"]]["pre_trained_sr"] != "None":
        print("Loading from existing Sr_FCN and copying weights to continue....")
        checkpoint = torch.load(config[config["train_dataset"]]["pre_trained_sr"])
        model.load_state_dict(checkpoint, strict=False)
    if args.resume is not None:
        print("Loading from existing model and copying weights to continue....")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint, strict=False)


    # Setting up loss functions.
    if config[config["train_dataset"]]["loss_type"] == "L1":
        criterion = torch.nn.L1Loss()

    elif config[config["train_dataset"]]["loss_type"] == "MSE":
        criterion = torch.nn.MSELoss()

    else:
        exit("Undefined loss type")


    # Setting up tensorboard and copy .json file to save directory.
    PATH = "./" + config["experim_name"] + "/" + config["train_dataset"] + "/" + config["upscale_method"]
    ensure_dir(PATH + "/")
    writer = SummaryWriter(log_dir=PATH)
    shutil.copy2(args.config, PATH)

    # Print model to text file
    original_stdout = sys.stdout
    with open(PATH + "/" + "model_summary.txt", 'w+') as f:
        sys.stdout = f
        print(f'\n{model}\n')
        sys.stdout = original_stdout

    # Main loop.
    best_psnr = 0.0
    for epoch in range(start_epoch, total_epochs):
        scheduler_ae.step(epoch)
        scheduler_rest.step(epoch)
        print("\nTraining Epoch: %d" % epoch)
        #print(torch.cuda.memory_summary())
        # train_vae(epoch)
        train_ae(epoch)
        if epoch % config["trainer"]["test_freq"] == 0:
            print("\nTesting Epoch: %d" % epoch)
            # image_dict, pred_dic, metrics = test_vae(epoch)
            metrics = test_ae(epoch)
            # Saving the best model
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]

                # Saving best performance metrics
                torch.save(model.state_dict(), PATH + "/" + "best_model.pth")
                with open(PATH + "/" + "best_metrics.json", "w+") as outfile:
                    json.dump(metrics, outfile)



