import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from scipy.signal import correlate2d as correlate
from scipy.signal import general_gaussian
from scipy import ndimage




def GetPSD1D(psd2D):
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.mean(psd2D, r, index=np.arange(0, wc))

    return psd1D

def RMS_WF(array):
    rms = np.sqrt(np.sum((1/(2*np.pi)*array)**2))
    rms = 1./(rms)
    return rms


def Apod(im, size,power, sigma):
    av = im.mean()
    im = im-av
    gen_gaussian = general_gaussian(size, power, sigma)
    window = np.outer(gen_gaussian, gen_gaussian)
    apod = im*window
    apod = apod+av
    return apod

def apo2d(masi,perc):
 s = masi.shape
 edge = 100./perc
 mean = np.mean(masi)
 masi = masi-mean
 xmask = np.ones(s[1])
 ymask = np.ones(s[0])
 smooth_x = np.int(s[1]/edge)
 smooth_y = np.int(s[0]/edge)

 for i in range(0,smooth_x):
    xmask[i] = (1.-np.cos(np.pi*np.float(i)/np.float(smooth_x)))/2.
    ymask[i] = (1.-np.cos(np.pi*np.float(i)/np.float(smooth_y)))/2.
    
 xmask[s[1] - smooth_x:s[1]] = (xmask[0:smooth_x])[::-1]
 ymask[s[0] - smooth_y:s[0]] = (ymask[0:smooth_y])[::-1]

#mask_x = np.outer(xmask,xmask)
#mask_y = np.outer(ymask,ymask)
 for i in range(0,s[1]):
    masi[:,i] = masi[:,i]*xmask[i]
 for i in range(0,s[0]):
    masi[i,:] = masi[i,:]*ymask[i]
 masi = masi+mean
 return masi


def strehl(rms):
    return np.exp(-2*(np.pi*rms**2))

def imreg(im0,imk):
    import imreg_dft as ird
    from image_registration import chi2_shift
    from image_registration.fft_tools import shift
    xoff, yoff, exoff, eyoff = chi2_shift(im0,imk)
    timg = ird.transform_img(imk, tvec=np.array([-yoff,-xoff]))
    return timg

def RMS_cont(data):
    return data.std()/data.mean()

def noise(im):
    from skimage.restoration import estimate_sigma
    s = estimate_sigma(im)
    return s


def plot_zernike(coeff): 
    n = coeff.shape[0] 
    index = np.arange(n) 
    fig = plt.figure(figsize=(9, 6), dpi=80) 
    width = 0.4 
    for i in index:  
        #xticklist.append('Z'+str(i+4))  
        barfigure = plt.bar(index, coeff/(2*np.pi), width,color = '#2E9AFE',edgecolor = '#2E9AFE')  
        plt.xticks(np.arange(1, 11, step=1)) 
        plt.xlabel('Zernike Polynomials',fontsize=18) 
        plt.ylabel('Coefficient [$\lambda$]',fontsize=18) 
        plt.title('Zernike Polynomials Coefficients',fontsize=18)

def prepare_patches(d,Del,Im0,Imk):
    n = d.shape[0]
    upper = 1700-Del
    lower = 300
    Nx = np.arange(lower,upper,Del)
    Ny = np.arange(lower,upper,Del)
    i_max = np.floor((upper-lower)/Del)+1
    patches = np.zeros((int(i_max**2),Del,Del,n))
    #output_WF =np.zeros((int(i_max**2),Del,Del))
    #output_mtf = np.zeros((int(i_max**2),Del,Del))
    k=0
    for n1 in Nx :
       for n2 in Ny:
        patches[k,:,:,0]=Im0[n2:n2+Del,n1:n1+Del]
        patches[k,:,:,1]=Imk[n2:n2+Del,n1:n1+Del]
        k = k+1
    return patches


def stitch_patches(results,Del):
    data1 = [r[0] for r in results]
    data2 = [r[1] for r in results]
    upper = 1700-Del
    lower = 300
    Nx = np.arange(lower,upper,Del)
    Ny = np.arange(lower,upper,Del)
    i_max = np.floor((upper-lower)/Del)+1
    k = 0
    if len(data1)==(np.floor((upper-lower)/Del)+1)**2:

      st_wf = np.zeros((2048,2048))
      st_mtf = np.zeros((2048,2048))
    else:
        raise TypeError('Check dimensions!')

    for n1 in Nx :
         for n2 in Ny:
             st_wf[n2:n2+Del,n1:n1+Del] = data1[k]
             st_mtf[n2:n2+Del,n1:n1+Del] = data2[k]
             k=k+1
    return st_wf,st_mtf
    

   



