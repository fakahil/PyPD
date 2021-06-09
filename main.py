## This is the main code. The main parameters needed to run the code should be input here. This includes: detector and telescope parameters, number of Zernike modes, noise parameters, images to be restored 

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

import aperture
import noise
import telescope
import tools
import wavefront
import zernike
import PD

## '''''''telescope parametes

## Aperture diameter (main aperture) in mm
D = 140

## wavelength of operation in mm
lam = 617.3*10**(-6)

## effective focal length in mm
f = 4125.3 

## introduced wavelength shift (in units of wavelength)
del_z = 0.5



## '''''''Detector parameters

## plate scale of detector (arcsec/pixel)
pix = 0.5

## Size of the detector (number of pixels)
size_detect = 2048

## Noise cut_off frequency (in units of 1/pixel with maximum cutoff being 0.5 pixel^-1)
cut_off = 0.5

## Regularization parameter for the Wiener Wilter (to be tested)
reg = 1e-10

## '''''''''''''''''''Focused-defocused data
## 
path_images = "/home/fatima/Desktop/solar_orbiter_project/commissioning/phi-5/"
im0 = pyfits.getdata(path_images+'solo_L0_phi-hrt-ilam_20200420T132834_V202004201547C_0024160022000.fits')[0,:,:]
imk = pyfits.getdata(path_images+'solo_L0_phi-hrt-ilam_20200420T132834_V202004201547C_0024160022000.fits')[1,:,:]

dark = pyfits.getdata(path_images+'dark_kinga.fits')
flat_field = pyfits.getdata(path_images+'solo_L0_phi-hrt-ilam_20200417T234528_V202004251106C_0024150029000.fits')[20,:,:]

im0 = (im0/20./256.- dark/256./20.)/(flat_field/256./150. - dark/256./20.)
imk = (imk/20./256.- dark/256./20.)/(flat_field/256./150. - dark/256./20.)
 
## Trimming to sub-boxes
x1 = 800
x2 = 900
y1 = 800
y2 =900
size = x2-x1
#percent of apodization 
ap =   10 

## Aligning focused-defocused data
im0 = im0[y1:y2,x1:x2]
imk = imk[y1:y2,x1:x2]
imk = tools.imreg(im0,imk)

## apodizing focused-defocused data
im0 = tools.apo2d(im0,ap)
imk = tools.apo2d(imk, ap)


## Zernike modes to be fitted

co_num = 3


rpupil = telescope.pupil_size(D,lam,pix,size)

Mask = aperture.mask(rpupil,size)

d0,dk = PD.FT(im0,imk)

#freq_mask =noise.noise_mask(size,cut_off)
#N0 =  noise.Noise_d0(d0, freq_mask)
noise_filter = fftshift(noise.noise_mask_high(size,cut_off))
#M_gamma = noise_mask(size,0.5)
gam = 1#noise.Gamma(d0,dk,M_gamma)



def Minimize(coefficients):
 
 A_f = wavefront.pupil_foc(coefficients,size,rpupil,co_num)
 A_def = wavefront.pupil_defocus(coefficients,size,del_z,rpupil,co_num)
 
 psf_foc = wavefront.PSF(Mask,A_f,False) 
 psf_defoc = wavefront.PSF(Mask,A_def,False)
 t0 = wavefront.OTF(psf_foc)
 tk = wavefront.OTF(psf_defoc)
 
 q,q2 = PD.Q_matrix(t0,tk,reg,gam)
  
 #noise_filter = sch_filter(N0, t0, tk, d0, dk,reg)
 F_m = PD.F_M(q2,d0, dk,t0,tk,noise_filter,gam)

 E_metric = PD.Error_metric(t0,tk,d0,dk,q,noise_filter)
 L_m = PD.L_M(E_metric,size)
 return L_m

from scipy.optimize import minimize
def main():
 start_time = time.time()

 p0 =   np.zeros(co_num)
 Minimize_partial = partial(Minimize)

 mini = minimize(Minimize_partial,p0,method='L-BFGS-B')

 end_time = time.time()
 print("Run time = {}".format(end_time - start_time))
 print(mini.x)

main()

