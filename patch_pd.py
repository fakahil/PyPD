import time
import imreg_dft
import pyfits
import tools
import aperture
import PD
import noise
import wavefront
import deconvolution
import cost_func
from scipy.optimize import minimize, minimize_scalar
import astropy 
from astropy.io import fits
import argparse
import telescope
from telescope import *

from tools import imreg, apo2d
from aperture import *
from PD import *
from noise import *
from wavefront import *
from deconvolution import *
from cost_func import *


class patch_pd(object):
    def __init__(self,pd_data,Del,co_num,output_wf,output_mtf):
      

     self.data= fits.getdata(pd_data)
     self.Del = Del
     self.co_num = co_num
     self.output_wf = output_wf
     self.output_mtf = output_mtf


     self.mean_im0 = self.data[0,500:1000,500:1000].mean()
     self.mean_imk = self.data[1,500:1000,500:1000].mean()

     self.Im0 = self.data[0,:,:]/self.mean_im0
     self.Imk = self.data[1,:,:]/self.mean_imk

     self.output_WF = np.zeros((2048,2048))
     self.output_mtf = np.zeros((2048,2048))



    def fit_patch(self):

      upper = 1700
      Nx = np.arange(300,upper,self.Del)
      Ny = np.arange(300,upper,self.Del)
      for n1 in Nx :
        for n2 in Ny:

            print('fitting the area in:'+str(n1)+','+str(n2))
            im0 = self.Im0[n2:n2+self.Del,n1:n1+self.Del]
            imk  = self.Imk[n2:n2+self.Del,n1:n1+self.Del]
            im0 = im0/im0.mean()
            imk = imk/imk.mean()
            imk = imreg(im0,imk)


           
            im0 = apo2d(im0,10)
            imk = apo2d(imk,10)
        
            d0,dk = FT(im0,imk)
            gam =1# Gamma(d0,dk,M_gamma)

            p0 =   np.zeros(self.co_num)
            fit = cost_func(self.Del,0.5,1e-10,10,self.co_num,0.5,617.3e-6, 140,4125.3,0.5)
            Mask = aperture.mask_pupil(fit.telescope.pupil_size(),fit.size)
            noise_temp = noise_mask_high(fit.size,fit.cut_off)
            noise_filter = fftshift(noise_temp)
            def Minimise(coefficients):
                  A_f = wavefront.pupil_foc(coefficients,fit.size,fit.telescope.pupil_size(),self.co_num)
                  A_def = wavefront.pupil_defocus(coefficients,fit.size,fit.del_z,fit.telescope.pupil_size(),self.co_num)
                  psf_foc = wavefront.PSF(Mask,A_f,False) 
                  psf_defoc = wavefront.PSF(Mask,A_def,False)
                  t0 = wavefront.OTF(psf_foc)
                  tk = wavefront.OTF(psf_defoc) 

                  q,q2 = PD.Q_matrix(t0,tk,fit.reg,gam)
             

                  F_m = PD.F_M(q2,d0, dk,t0,tk,noise_filter,gam)

                  E_metric = PD.Error_metric(t0,tk,d0,dk,q,noise_filter)
                  L_m = PD.L_M(E_metric,fit.size)
                  return L_m

       
            Minimise_partial = partial(Minimise)
            mini = scipy.optimize.minimize(Minimise_partial,p0,method= 'L-BFGS-B')
            
            result = fit.Minimize_res(mini.x)
            self.output_WF[n2:n2+self.Del,n1:n1+self.Del] = result[1]
            self.output_mtf[n2:n2+self.Del,n1:n1+self.Del] = MTF(fftshift(result[0]))
            hdu = fits.PrimaryHDU(self.output_WF)
            hdu.writeto(self.output_wf,overwrite=True)

         
if (__name__ == '__main__'):
 parser = argparse.ArgumentParser(description='PD on sub-fields')
 parser.add_argument('-i','--input', help='input')
 parser.add_argument('-z','--Z', help='Zernike',default=10)
 parser.add_argument('-d','--Del', help='Del',default=265)
 parser.add_argument('-ow','--ow', help='output_WFE')
 parser.add_argument('-om','--om', help='output MTF')
 parsed = vars(parser.parse_args())
 st = patch_pd(pd_data='{0}'.format(parsed['input']),Del=int(parsed['Del']),co_num=int(parsed['Z']),output_wf='{0}'.format(parsed['ow']),output_mtf='{0}'.format(parsed['om']))
 st.fit_patch()
   
