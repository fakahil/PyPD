
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
import tools
import aperture
import PD
import noise
import wavefront
from scipy.optimize import minimize, minimize_scalar
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

def Wienerfilter(img,t0,reg,cut_off,size):
    noise_filter = fftshift(noise.noise_mask_high(size,cut_off))

    im0 = tools.apo2d(img,ap)
    d0 = fft2(im0)
    scene = noise_filter*d0*(np.conj(t0)/(np.abs(t0)**2+reg))
    scene2 = ifft2(scene).real  
    return scene2



