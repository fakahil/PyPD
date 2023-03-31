
import numpy as np
import scipy
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from functools import partial


import pyfits



## function to compute the FT of focused and defocused image
def FT(im0, imk):
    d0 = fft2(im0)
    dk = fft2(imk)
    return d0, dk
    

## function to define the Q matrix (Eq. 6 of L\"odfahl & scharmer. 1994)
def Q_matrix(t0,tk,reg,gamma):
    tmp = np.abs(t0)**2 + gamma*np.abs(tk)**2 +reg
    q = 1./(np.sqrt(tmp))
    q2 = q*q
    return q, q2



#
## function to compute the optimized object (Eq. 5 of L\"odfahl & scharmer.1994)
def F_M(q2,d0,dk,t0,tk,filter,gamma):
    F_M = filter*q2*(d0*np.conj(t0) + gamma*dk*np.conj(tk))
    return F_M




## function to define the error metric to be minimized (Eq. 9 of  L\"odfahl & scharmer.1994)
def Error_metric(t0,tk,d0,dk,q,filter):
    ef = filter*(dk*t0 - d0*tk)
    ef = q*ef
    EF = fft2((ef))
    EF = EF.real
    EF = EF-EF.mean()
    return  EF



## 
def L_M(EF,size):
    L_m = np.sum(np.abs(EF)**2)/(size**2)
    return L_m

