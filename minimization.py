
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
import deconvolution
from scipy.optimize import minimize, minimize_scalar


#from . import detector
#from .detector import *
from telescope import *

from tools import imreg, apo2d
from aperture import *
from PD import *
from noise import *
from wavefront import *
from deconvolution import *

class minimization(object):

   def __init__(self,foc,defoc,size,lam, diameter,focal_length,platescale,cut_off,reg,ap,x1,x2,y1,y2,co_num,del_z):
      self.foc = foc
      self.defoc = defoc
      self.cut_off = cut_off
      self.reg = reg
      self.size = size
      self.ap = ap
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
      self.reg = reg
      self.telescope = Telescope(self.lam,self.diameter,self.focal_length,'HRT',self.platescale,self.size)




   def fit(self,m = 'L-BFGS-B'):
      im0 = self.foc[self.y1:self.y2,self.x1:self.x2]
      imk = self.defoc[self.y1:self.y2,self.x1:self.x2]
      imk = tools.imreg(im0,imk)
      im0 = tools.apo2d(im0,self.ap)
      imk = tools.apo2d(imk, self.ap)
      rpupil = self.telescope.pupil_size()
      Mask = aperture.mask_pupil(rpupil,self.size)
      
      d0,dk = PD.FT(im0,imk)
      
      noise_filter = fftshift(noise.noise_mask_high(self.size,self.cut_off))
      M_gamma = noise.noise_mask(self.size,self.cut_off)
      gam = noise.Gamma(d0,dk,M_gamma)
      p0 =   np.zeros(self.co_num)

      def Minimise(coefficients):
       A_f = wavefront.pupil_foc(coefficients,self.size,self.telescope.pupil_size(),self.co_num)
       A_def = wavefront.pupil_defocus(coefficients,self.size,self.del_z,self.telescope.pupil_size(),self.co_num)
       psf_foc = wavefront.PSF(Mask,A_f,False) 
       psf_defoc = wavefront.PSF(Mask,A_def,False)
       t0 = wavefront.OTF(psf_foc)
       tk = wavefront.OTF(psf_defoc) 

       q,q2 = PD.Q_matrix(t0,tk,self.reg,gam)
  
       #noise_filter = sch_filter(N0, t0, tk, d0, dk,reg)
       F_m = PD.F_M(q2,d0, dk,t0,tk,noise_filter,gam)

       E_metric = PD.Error_metric(t0,tk,d0,dk,q,noise_filter)
       L_m = PD.L_M(E_metric,self.size)
       return L_m

      Minimise_partial = partial(Minimise)
      mini = scipy.optimize.minimize(Minimise_partial,p0,method=m)
      return mini.x


   def plot_results(self,Z,wav,modulation,azimuthal):

    if len(Z)!=self.co_num:
      self.co_num = len(Z)
     
      
      A_f = wavefront.pupil_foc(Z,self.size,self.telescope.pupil_size(),self.co_num) 
      rpupil = self.telescope.pupil_size()
      Mask = aperture.mask_pupil(rpupil,self.size)
      psf_foc = wavefront.PSF(Mask,A_f,False) 
      t0 = wavefront.OTF(psf_foc)
      ph =wavefront.phase(Z, self.telescope.pupil_size(),self.co_num)
     
      if wav=='True':
      #def plot_wavefront(self):
        fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(1,1,1)

	im = ax.imshow(ph/(2*np.pi), origin='lower',cmap='gray')
        ax.set_xlabel('[Pixels]',fontsize=18)
	ax.set_ylabel('[Pixels]',fontsize=18)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size=0.15, pad=0.05)

	cbar = plt.colorbar(im, cax=cax,orientation='vertical')#,ticks=np.arange(0.4,0.9,0.1))
        cbar.set_label('WF error HRT [$\lambda$]',fontsize=20)
	plt.show() 
      #else:
        #pass

      #def plot_mtf(self):
      if modulation=='True':
        fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(1,1,1)
        im = ax.imshow(wavefront.MTF(fftshift(t0)), origin='lower',cmap='gray')
        ax.set_xlabel('[Pixels]',fontsize=18)
	ax.set_ylabel('[Pixels]',fontsize=18)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size=0.15, pad=0.05)

	cbar = plt.colorbar(im, cax=cax,orientation='vertical')#,ticks=np.arange(0.4,0.9,0.1))
        cbar.set_label('MTF',fontsize=20)

        plt.show()
      #else:
        #pass

      if azimuthal =='True':
      #def plot_azimuthal_mtf(self):
        az = tools.GetPSD1D(wavefront.MTF(fftshift(t0)))
        freq=np.linspace(0,0.5,int(self.size/2))
        freq_c_hrt = self.diameter/(self.lam*self.focal_length*100)
        phi_hrt = np.arccos(freq/freq_c_hrt)
        MTF_p_hrt = (2/np.pi)*(phi_hrt - (np.cos(phi_hrt))*np.sin(phi_hrt))
        fig = plt.figure(figsize=(10,10))
        plt.xlabel('Freq (1/pixel)',fontsize=22)
        plt.ylabel('MTF',fontsize=22)
        plt.plot(freq,MTF_p_hrt,label='Theoretical MTF')
        plt.plot(freq,az,label='Observed MTF')
        plt.xlim(0,0.5)
        plt.legend()
        plt.show()

   
   def restored_scene(self,Z, to_clean,filter,iterations_RL):

      A_f = wavefront.pupil_foc(Z,self.size,self.telescope.pupil_size(),self.co_num) 
      rpupil = self.telescope.pupil_size()
      Mask = aperture.mask_pupil(rpupil,self.size)
      psf_foc = wavefront.PSF(Mask,A_f,False) 
      t0 = wavefront.OTF(psf_foc)
      if to_clean.shape != t0.shape:
         raise ValueError('Image and PSF do not have the same dimensions')
      elif to_clean.shape == t0.shape:

        if filter == 'Wiener':
         restored = deconvolution.Wienerfilter(to_clean,t0,self.reg,self.cut_off,self.size) 
         return restored
        if filter == 'richardsonLucy':
         restored = deconvolution.richardsonLucy(to_clean, psf_foc, iterations_RL)
         return restored



       
      
      
