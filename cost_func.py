import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2


import tools
import aperture
import PD
import noise
import wavefront
import telescope
from telescope import *
from noise import *
from wavefront import *
from tools import *
from PD import *


class cost_func:
   def __init__(self,size,cut_off,reg,ap,co_num,del_z,lam, diameter,focal_length,platescale):
    
      self.cut_off = cut_off
      self.reg = reg
      self.size = size

      self.co_num = co_num
    
      self.del_z = del_z
      self.lam = lam
      self.diameter = diameter
      self.focal_length = focal_length
      self.platescale = platescale
      self.telescope = Telescope(self.lam,self.diameter,self.focal_length,'HRT',self.platescale,self.size)





   def Minimize_res(self,coefficients):

       noise_filter = fftshift(noise_mask_high(self.size,self.cut_off))
       
       ph = wavefront.phase_embed(coefficients,self.telescope.pupil_size(),self.size,self.co_num)
       A_f = wavefront.pupil_foc(coefficients,self.size,self.telescope.pupil_size(),self.co_num)


       psf_foc = wavefront.PSF( aperture.mask_pupil(self.telescope.pupil_size(),self.size),A_f,False) 

       t0 = wavefront.OTF(psf_foc)
       
          
       return t0,ph
  
