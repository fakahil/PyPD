
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

import argparse
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

   def __init__(self,foc_defoc,lam, diameter,focal_length,platescale,cut_off,reg,ap,x1,x2,y1,y2,co_num,del_z,output,filterr):
      self.data = fits.getdata(foc_defoc)


      self.foc =  self.data[0,:,:]
      self.defoc = self.data[1,:,:]
      self.cut_off = cut_off
      self.reg = reg
      self.size =self.y2-self.y1
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
      self.output = output
      self.filterr = filterr
     


   def fit(self,m = 'L-BFGS-B'):
      im0 = self.foc[self.y1:self.y2,self.x1:self.x2]
      imk = self.defoc[self.y1:self.y2,self.x1:self.x2]
      imk = tools.imreg(im0,imk)
      im0 = tools.apo2d(im0,self.ap)
      imk = tools.apo2d(imk, self.ap)
      rpupil = self.telescope.pupil_size()
      Mask = aperture.mask_pupil(rpupil,self.size)
      
      d0,dk = PD.FT(im0,imk)
      noise_temp = noise_mask_high(self.size,self.cut_off)
      noise_filter = fftshift(noise_temp)

      M_gamma = noise_mask(self.size,self.cut_off)
      gam = Gamma(d0,dk,M_gamma)
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


   def plot_results(self,Z):

     
      
      A_f = wavefront.pupil_foc(Z,self.size,self.telescope.pupil_size(),self.co_num) 
      rpupil = self.telescope.pupil_size()
      Mask = aperture.mask_pupil(rpupil,self.size)
      psf_foc = wavefront.PSF(Mask,A_f,False) 
      t0 = wavefront.OTF(psf_foc)
      ph =wavefront.phase(Z, self.telescope.pupil_size(),self.co_num)
     
     
      fig = plt.figure(figsize=(20,20))
      ax1 = fig.add_subplot(1,3,1)
      im1 = ax1.imshow(ph/(2*np.pi), origin='lower',cmap='gray')
      ax1.set_xlabel('[Pixels]',fontsize=18)
      ax1.set_ylabel('[Pixels]',fontsize=18)
      divider1 = make_axes_locatable(ax1)
      cax1 = divider1.append_axes("right", size=0.15, pad=0.05)
      cbar1 = plt.colorbar(im1, cax=cax1,orientation='vertical')#,ticks=np.arange(0.4,0.9,0.1))
      ax1.set_title('WF error HRT [$\lambda$]',fontsize=20)

      ax2 = fig.add_subplot(1,3,2)
      im2 = ax2.imshow(wavefront.MTF(fftshift(t0)), origin='lower',cmap='gray')
      ax2.set_xlabel('[Pixels]',fontsize=18)
      ax2.set_ylabel('[Pixels]',fontsize=18)
      divider2 = make_axes_locatable(ax2)
      cax2 = divider2.append_axes("right", size=0.15, pad=0.05)
      cbar2 = plt.colorbar(im2, cax=cax2,orientation='vertical')#,ticks=np.arange(0.4,0.9,0.1))
      ax2.set_title('MTF',fontsize=20)

 
      az = tools.GetPSD1D(wavefront.MTF(fftshift(t0)))
      freq=np.linspace(0,0.5,int(self.size/2))
      freq_c_hrt = self.diameter/(self.lam*self.focal_length*100)
      phi_hrt = np.arccos(freq/freq_c_hrt)
      MTF_p_hrt = (2/np.pi)*(phi_hrt - (np.cos(phi_hrt))*np.sin(phi_hrt))
      ax3 = fig.add_subplot(1,3,3)
      ax3.set_xlabel('Freq (1/pixel)',fontsize=22)
      ax3.set_ylabel('MTF',fontsize=22)
      ax3.plot(freq,MTF_p_hrt,label='Theoretical MTF')
      ax3.plot(freq,az,label='Observed MTF')
      ax3.set_xlim(0,0.5)
      plt.legend()
      plt.subplots_adjust(hspace=0.1, wspace=0.6)

      plt.savefig('resutls.png')

   
   def restored_scene(self,Z,iterations_RL=10):
      to_clean = self.foc[self.y1:self.y2,self.x1:self.x2]
      A_f = wavefront.pupil_foc(Z,self.size,self.telescope.pupil_size(),self.co_num) 
      rpupil = self.telescope.pupil_size()
      Mask = aperture.mask_pupil(rpupil,self.size)
      psf_foc = wavefront.PSF(Mask,A_f,False) 
      t0 = wavefront.OTF(psf_foc)
      if to_clean.shape != t0.shape:
         raise ValueError('Image and PSF do not have the same dimensions')
      elif to_clean.shape == t0.shape:

        if self.filterr == 'Wiener':
         restored =Wienerfilter(to_clean,t0,self.reg,self.cut_off,self.size,self.ap) 
  

        if self.filterr == 'richardsonLucy':
         restored = richardsonLucy(to_clean, psf_foc, iterations_RL)

        print("Saving data...")
        hdu = fits.PrimaryHDU(restored)
        hdu.writeto(self.output,overwrite=True)
        '''
        import os.path
        if os.path.exists(self.output):
            os.system('rm {0}'.format(self.output))
            print('Overwriting...')
            hdu.writeto(self.output,overwrite=True)

        '''


if (__name__ == '__main__'):



 parser = argparse.ArgumentParser(description='Retrieving wavefront error')
 parser.add_argument('-i','--input', help='input')
 parser.add_argument('-o','--out', help='out')
 parser.add_argument('-w','--wavelength', help='wavelength',default=617.3e-6)
 parser.add_argument('-a','--aperture', help='aperture', default=140)
 parser.add_argument('-f','--focal_length', help='focal_length',default=4125.3)
 parser.add_argument('-p','--plate_scale', help='plate_scale',default=0.5)
 parser.add_argument('-c','--cut_off', help='cut_off',default=0.5)
 parser.add_argument('-r','--reg', help='reg',default=1e-10)
 parser.add_argument('-ap','--apod', help='apod',default=10)
 parser.add_argument('-x1','--x1', help='x1')
 parser.add_argument('-x2','--x2', help='x2')
 parser.add_argument('-y1','--y1', help='y1')
 parser.add_argument('-y2','--y2', help='y2')
 parser.add_argument('-z','--Z', help='Z',default=10)
 parser.add_argument('-del','--del', help='del',default=0.5)
 parser.add_argument('-fl','--filter', help='filter', choices=['richardsonLucy', 'Wiener'], default='Wiener')

 parsed = vars(parser.parse_args())



 res = minimization(foc_defoc='{0}'.format(parsed['input']), lam=float(parsed['wavelength']),diameter=float(parsed['aperture']),focal_length=float(parsed['focal_length']),
 platescale=float(parsed['plate_scale']),cut_off=float(parsed['cut_off']),reg=float(parsed['reg']),ap=int(parsed['apod']),x1=int(parsed['x1']),x2=int(parsed['x2']),y1=int(parsed['y1']),y2=int(parsed['y2']),
 co_num=int(parsed['Z']),del_z=float(parsed['del']),output='{0}'.format(parsed['out']),filterr=parsed['filter'])

 Z = res.fit() 
 print(Z)
 tools.plot_zernike(Z)
 res.plot_results(Z)
 res.restored_scene(Z,10)


      
