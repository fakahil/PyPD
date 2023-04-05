
import numpy as np

import scipy
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from astropy.io import fits
import imreg_dft

import tools
import aperture
import PD
import noise
import wavefront
import wavefront
from wavefront import *
import tools 
from tools import *
from noise import *

def ft2(g):
    G = fftshift(fft2(ifftshift(g))) 
    return G

def ift2(G):

    numPixels = G.shape[0]

    g = fftshift(ifft2(ifftshift(G))) 
    return g


def conv2(g1, g2):

    G1 = ft2(g1)
    G2 = ft2(g2)
    G_out = G1*G2

    numPixels = g1.shape[0]    
   
    g_out = ift2(G_out)
    return g_out


def richardsonLucy(img, psf, iterations=100):
    f = img
    blur_psf = np.matrix(psf)

    psf_mirror = blur_psf.T

    for _ in range(iterations):

        f = f * conv2(img / conv2(f, blur_psf), psf_mirror)

    return np.abs(f)

def Wienerfilter(img,t0,reg,cut_off,size,ap):
    temp = noise_mask_high(size,cut_off)
    noise_filter = fftshift(temp)

    im0 = tools.apo2d(img,ap)
    d0 = fft2(im0)
    scene = noise_filter*d0*(np.conj(t0)/(np.abs(t0)**2+reg))
    scene2 = ifft2(scene).real  
    return scene2



