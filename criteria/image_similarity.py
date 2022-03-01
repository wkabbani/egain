import math
import numpy as np
import cv2
import os
import json
import image_similarity_measures
from sys import argv
from image_similarity_measures.quality_metrics import fsim, issm, sre, uiq
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_vifp(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return vifp(img2, img1)

def calculate_msssim(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return msssim(img2, img1)

def calculate_sam(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return sam(img2, img1)

def calculate_rase(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return rase(img2, img1)

def calculate_scc(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return scc(img2, img1)

def calculate_ergas(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return ergas(img2, img1)

def calculate_uqi(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return uqi(img2, img1)

def calculate_rmse(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return rmse(img2, img1)

def calculate_mse(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return mse(img2, img1)

def calculate_fsim(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return fsim(img2, img1)

def calculate_issm(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return issm(img2, img1)

def calculate_sre(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return sre(img2, img1)

def calculate_uiq(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return uiq(img2, img1)

def calculate_psnr(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return psnr(img2, img1, data_range=img2.max() - img2.min())

def calculate_ssim(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))

    return ssim(img2, img1, data_range=img2.max() - img2.min(), multichannel=True)