import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from scipy.signal import correlate2d as correlate
from scipy.signal import general_gaussian
from astropy.io import fits
from scipy import ndimage
from functools import partial
import time
import imreg_dft
import pyfits


# compute the gamma parameter
def Gamma(d0,dk,mask):
    d0 = np.abs(d0)**2
    dk = np.abs(dk)**2
    d0 = fftshift(d0)
    dk = fftshift(dk)
    denom = np.sum(mask)
    sigma_0 = np.sum(mask*d0)/denom
    sigma_k = np.sum(mask*dk)/denom
    gamma = sigma_0/sigma_k
    return gamma   


# mask frequencies below a frequency threshold (to make the noise image)
def noise_mask(size,cut_off):
   X = np.linspace(-0.5,0.5,size)
   x,y = np.meshgrid(X,X)
   mask = np.ones((size,size))
   m = x * x + y * y <= cut_off**2
   mask[m] = 0
   return mask



# to mask frequencies above a frequency threshold
def noise_mask_high(size,cut_off):
   X = np.linspace(-0.5,0.5,size)
   x,y = np.meshgrid(X,X)
   mask = np.zeros((size,size))
   m = x * x + y * y <= cut_off**2
   mask[m] = 1
   return mask




def Noise_d0(im0, freq_mask):
    pd = fft2(im0)
    pd = np.abs(pd)**2
    pd = fftshift(pd)
    pd_m = pd*freq_mask
    pd_m = ifftshift(pd_m)
    return pd_m




## Noise filter (Loedfahl and Berger 1998)
def sch_filter(noise_im,t0,tk,d0,dk,reg):
    
    ab_noise = np.abs(noise_im)**2
    filter = ab_noise*(np.abs(t0)**2 + np.abs(tk)**2)/(np.abs(d0*np.conj(t0) + dk*np.conj(tk))**2 + reg)
    filter = 1- filter
    filter_2 = filter
    filter_2[filter<0.2] = 0
    filter_2[filter>1] = 1
    return filter_2

