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
import zernike

def phase(coefficients,rpupil, co_num):
 r = 1
 x = np.linspace(-r, r, 2*rpupil)
 y = np.linspace(-r, r, 2*rpupil)

 [X,Y] = np.meshgrid(x,y) 
 R = np.sqrt(X**2+Y**2)
 theta = np.arctan2(Y, X)
    
 Z = zernike.Zernike_polar(coefficients,R,theta, co_num)
 Z[R>1] = 0
 return Z


def phase_aberr(del_z,rpupil):
 alpha_4 = (del_z*np.pi)/np.sqrt(3)
 r = 1
 x = np.linspace(-r, r, 2*rpupil)
 y = np.linspace(-r, r, 2*rpupil)
 [X,Y] = np.meshgrid(x,y)
 #ph_defocus = -(np.pi*del_z*(X**2+Y**2)*D**2)/(4*lam*f**2)
 R = np.sqrt(X**2 + Y**2)
 ph_defocus = alpha_4  * np.sqrt(3)*(2*R**2-1)
 ph_defocus[R>1] = 0
 return ph_defocus


# Embed the phase in the center of a quadratic map of dimensions sizexsize
def phase_embed(coefficients,rpupil,size,co_num):
    ph =  phase(coefficients,rpupil)
    A = np.zeros([size,size])
    A[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1]= phase(coefficients,rpupil,co_num)
    return A


## function for making the complex pupil function from Zernike
def pupil_foc(coefficients,size,rpupil,co_num):

    A = np.zeros([size,size])
    A[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1]= phase(coefficients,rpupil,co_num)
    aberr =  np.exp(1j*A)
    return aberr


## function for making the complex pupil function for the defocused wavefront
def pupil_defocus(coefficients,size,del_z,rpupil, co_num):
    A = np.zeros([size,size])
    A[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1] = phase_aberr(del_z,rpupil)+phase(coefficients,rpupil,co_num)
    
    aberr_defocus =  np.exp(1j*A)
    return  aberr_defocus




def PSF(mask,abbe,norm):
    ## making zero where the aberration is equal to 1 (the zero background)

 abbe_z = np.zeros((len(abbe),len(abbe)),dtype=np.complex)
 abbe_z = mask*abbe
 PSF = ifftshift(fft2(fftshift(abbe_z))) #from brandon
 PSF = (np.abs(PSF))**2 #or PSF*PSF.conjugate()
 if norm=='True':
  PSF = PSF/PSF.sum()
  return PSF
 else:
  return PSF

def OTF(psf):
    otf = ifftshift(psf)
    otf = fft2(otf)
    otf = otf/np.real(otf[0,0])

    #otf = otf/otf.max() # or otf_max = otf[size/2,size/2] if max is shifted to center
   
    return otf


def MTF(otf):
    mtf = np.abs(otf)
    return mtf



