import torch
import torch.nn as nn
import numpy as np
eps = 1e-10
import torch.nn.functional as F
import torchvision.transforms as transform
import math

# Cross-correlation matrix
def cross_correlation(H_fuse, H_ref):
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating mean value
    mean_fuse = torch.mean(H_fuse_reshaped, 1).unsqueeze(1)
    mean_ref = torch.mean(H_ref_reshaped, 1).unsqueeze(1)

    CC = torch.sum((H_fuse_reshaped - mean_fuse) * (H_ref_reshaped - mean_ref), 1) / torch.sqrt(
        torch.sum((H_fuse_reshaped - mean_fuse) ** 2, 1) * torch.sum((H_ref_reshaped - mean_ref) ** 2, 1))

    CC = torch.mean(CC)
    return CC

# Spectral-Angle-Mapper (SAM)
def SAM(H_fuse, H_ref):
    # Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)
    N_pixels = H_fuse_reshaped.shape[1]

    # Calculating inner product
    inner_prod = torch.nansum(H_fuse_reshaped * H_ref_reshaped, 0)
    fuse_norm = torch.nansum(H_fuse_reshaped ** 2, dim=0).sqrt()
    ref_norm = torch.nansum(H_ref_reshaped ** 2, dim=0).sqrt()

    # Calculating SAM
    SAM = torch.rad2deg(torch.nansum(torch.acos(inner_prod / (fuse_norm * ref_norm))) / N_pixels)
    return SAM

# Root-Mean-Squared Error (RMSE)
def RMSE(H_fuse, H_ref):
    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(-1)
    H_ref_reshaped = H_ref.view(-1)

    # Calculating RMSE
    RMSE = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped) ** 2) / H_fuse_reshaped.shape[0])
    return RMSE

# Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
def ERGAS(H_fuse, H_ref, beta):
    # Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)
    N_pixels = H_fuse_reshaped.shape[1]

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=1) / N_pixels)
    mu_ref = torch.mean(H_ref_reshaped, dim=1)

    # Calculating Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
    ERGAS = 100 * (1 / beta ) * torch.sqrt(torch.nansum(torch.div(rmse, mu_ref) ** 2) / N_spectral)
    return ERGAS

# Peak SNR (PSNR)
def PSNR(H_fuse, H_ref):
    # Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.sum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=1) / H_fuse_reshaped.shape[1])

    # Calculating max of H_ref for each band
    max_H_ref, _ = torch.max(H_ref_reshaped, dim=1)

    # Calculating PSNR
    PSNR = torch.nansum(10 * torch.log10(torch.div(max_H_ref, rmse) ** 2)) / N_spectral

    return PSNR

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance))* \
                      torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1)/ \
                              (2*variance)
                              )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel/torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

#Q_index
def Q_index(x, y):
    #x,y are tensor with a shape of (bands, pixels)
    N_pixels = x.shape[0]-1
    mean_x = torch.mean(x, 0)
    mean_y = torch.mean(y, 0)
    sigma_x = torch.sqrt(torch.sum((x-mean_x)**2)/N_pixels)
    sigma_y = torch.sqrt(torch.sum((y - mean_y)**2)/N_pixels)
    sigma_xy = torch.sum((x-mean_x)*(y-mean_y))/N_pixels
    q = 4*sigma_xy*mean_y*mean_x/((sigma_x**2+sigma_y**2)*(mean_x**2+mean_y**2))
    return q

#Spectral distortion index
def D_lambda(fused_hsi, lr_hsi, band_list):
    p=2
    selected_bandnum = len(band_list)
    bands = int(selected_bandnum*(selected_bandnum - 1)/2)
    Q1=torch.zeros(bands)
    Q2=torch.zeros(bands)
    N_spectral = fused_hsi.shape[1]
    fused_hsi_re = fused_hsi.view(N_spectral, -1)
    lr_hsi_re = lr_hsi.view(N_spectral, -1)
    index = 0
    for i in range(len(band_list)-1):
        for n in range(1, len(band_list)-i):
            Q1[index] = Q_index(fused_hsi_re[band_list[i]], fused_hsi_re[band_list[i+n]])
            Q2[index] = Q_index(lr_hsi_re[band_list[i]], lr_hsi_re[band_list[i+n]])
            index += 1
    d_lambda = (torch.nansum(((torch.abs(Q1-Q2))**p))*2/bands)**(1/p)
    return d_lambda

#Spatial distortion index
def D_s(fused_hsi, lr_hsi, pan, down_ratio):

    q=2
    N_spectral = fused_hsi.shape[1]
    fused_hsi_re = fused_hsi.view(N_spectral, -1)
    lr_hsi_re = lr_hsi.view(N_spectral, -1)
    pan_re = pan.reshape(-1)
    pan = pan.unsqueeze(0)

    sig = (1/(2*2.7725887/down_ratio**2))**0.5
    blur_layer = get_gaussian_kernel(7, sig, 1).cuda()
    degraded_pan = blur_layer(pan)
    degraded_pan = degraded_pan.squeeze()[::down_ratio, ::down_ratio]
    degraded_pan = degraded_pan.reshape(-1)
    Q1 = torch.zeros(N_spectral)
    Q2 = torch.zeros(N_spectral)
    for i in range(N_spectral):
        Q1[i] = Q_index(fused_hsi_re[i], pan_re)
        Q2[i] = Q_index(lr_hsi_re[i], degraded_pan)
    d_s = (torch.nansum((torch.abs(Q1-Q2)**q))/N_spectral)**(1/q)
    return d_s

#Quality index with no reference
def QNR(d_lambda, d_s):
    alpha = 1.0
    beta = 1.0
    qnr = ((1-d_lambda)**alpha)*((1-d_s)**beta)
    return qnr

def generate_band_list(N_bands, selected_num):
    arr = np.arange(N_bands)
    np.random.shuffle(arr)
    band_list = arr[0: selected_num]
    band_list = band_list.sort()

    return band_list

def average_D_lambda(fused_hsi, lr_hsi, selected_num, iters):
    N_bands = fused_hsi.shape[1]
    D_l = 0.0

    for i in range(iters):
        band_list = generate_band_list(N_bands, selected_num)
        D_l += D_lambda(fused_hsi, lr_hsi, band_list)

    D_l/=iters
    return D_l
