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
import os
import processing


class patch_pd(object):
    def __init__(self,pd_data,Del,co_num,output_wf,output_mtf,parallel=True):
      

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
     self.output_MTF = np.zeros((2048,2048))
     self.parallel = parallel
    #@staticmethod
    def run_pd(self,k):
       
     im0 = self.patch[k,:,:,0]
     imk  =self.patch[k,:,:,1]
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



     Minimize_partial = partial(Minimise)
     mini = minimize(Minimize_partial,p0,method='L-BFGS-B')
     result = fit.Minimize_res(mini.x)
     patch_wfe = result[1]
     patch_mtf = MTF(fftshift(result[0]))
     return patch_wfe,patch_mtf

    def fit_patch(self):

     upper = 1700
     Nx = np.arange(300,upper,self.Del)
     Ny = np.arange(300,upper,self.Del)
     if not self.parallel:
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
            self.output_MTF[n2:n2+self.Del,n1:n1+self.Del] = MTF(fftshift(result[0]))
            hdu = fits.PrimaryHDU(self.output_WF)
            hdu.writeto(self.output_wf,overwrite=True)
            hdu = fits.PrimaryHDU(self.output_mtf)
            hdu.writeto(self.output_mtf,overwrite=True)
     else:   
            self.patch = tools.prepare_patches(self.data,self.Del,self.Im0,self.Imk)
            n_workers = min(6, os.cpu_count())
            self.args_list = [i for i in range(len(self.patch))]
            self.results_parallel = list(processing.MP.simultaneous(self.run_pd, self.args_list, workers=n_workers))

    def plot_results(self,output):

         # change here the format of the output
         if not self.parallel:
            data_mtf = self.output_MTF
            data_wfe = self.output_WF

         if self.parallel:
            ## call here the stitching function
            data_mtf,data_wfe = tools.stitch_patches(self.results_parallel,self.Del)


         fig, ax = plt.subplots(1,2,figsize=(10,10))
        
         im=ax[1].imshow(data_mtf,vmin=0,vmax=1,origin='lower',cmap='gray')
         ax[1].set_title('MTF')
         divider = make_axes_locatable(ax[1])
         cax = divider.append_axes('right',pad=0.05,size=0.03)
         cbar1 = plt.colorbar(im,cax=cax)
         cbar1.ax.tick_params(labelsize=16)
         cbar1.set_label('MTF',fontsize=16)
         ax[1].set_xlabel('[Pixels]')
         major_ticks = np.arange(0, 2048,350)
         major_ticks_y = np.arange(0, 2048,350)
         ax[1].set_xticks(major_ticks)
         ax[1].set_yticks(major_ticks_y)
         ax[1].tick_params(labelsize=4)



         im2=ax[0].imshow(data_wfe/(2*np.pi),vmin=-0.5,vmax=1.4,origin='lower',cmap='gray')
         ax[0].set_xlabel('[Pixels]')
         ax[0].set_title('WF error[$\lambda$]')
         divider2 = make_axes_locatable(ax[0])
         cax2 = divider2.append_axes('right',pad=0.05,size=0.03) #size is the width of the color bar and pad is the fraction of the new axis
         cbar2=plt.colorbar(im2,cax=cax2)
         cbar2.ax.tick_params(labelsize=16)
         ax[0].tick_params(labelbottom=False,labelsize=16)
         plt.subplots_adjust(wspace=None, hspace=0.1)
         ax[0].set_xticks(major_ticks)
         ax[0].set_yticks(major_ticks_y)
         ax[0].set_ylabel('[Pixels]')
         #plt.subplots_adjust(wspace=None, hspace=-0.1)
         #plt.axis('off')
         plt.savefig(output,dpi=300)
         
if (__name__ == '__main__'):
 parser = argparse.ArgumentParser(description='PD on sub-fields')
 parser.add_argument('-i','--input', help='input')
 parser.add_argument('-z','--Z', help='Zernike',default=10)
 parser.add_argument('-d','--Del', help='Del',default=265)
 parser.add_argument('-ow','--ow', help='output_WFE')
 parser.add_argument('-om','--om', help='output MTF')
 parser.add_argument('-r','--res', help='results')
 parser.add_argument('-p','--parallel',choices=['True','False'],default=True)
 parsed = vars(parser.parse_args())
 st = patch_pd(pd_data='{0}'.format(parsed['input']),Del=int(parsed['Del']),co_num=int(parsed['Z']),output_wf='{0}'.format(parsed['ow']),output_mtf='{0}'.format(parsed['om']),parallel=bool(parsed['parallel']))
 st.fit_patch()
 st.plot_results(output='{0}'.format(parsed['res']))
   
