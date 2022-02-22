import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()

    img1 = img1.transpose((1,2,0))
    img2 = img2.transpose((1,2,0))
    # img1 and img2 have range [0, 255]
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    # mse = np.mean((img1 - img2)**2)
    # if mse == 0:
    #     return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr(img2, img1, data_range=img2.max() - img2.min())