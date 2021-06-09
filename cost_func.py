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


class fitting:
   def __init__(self,size,cut_off,reg,ap,x1,x2,y1,y2,co_num,del_z,lam, diameter,focal_length,platescale):
    
      self.cut_off = cut_off
      self.reg = reg
      self.size = size

      self.co_num = co_num
      self.x1 = x1
      self.x2 = x2
      self.y1 = y1
      self.y2 = y2
      self.del_z = del_z
      self.lam = lam
      self.diameter = diameter
      self.focal_length = focal_length
      self.platescale = platescale
      self.telescope = Telescope(self.lam,self.diameter,self.focal_length,'HRT',self.platescale,self.size)



  def Minimise(coefficients):
	      A_f = wavefront.pupil_foc(coefficients,self.size,rpupil,self.co_num)
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
   def run():
