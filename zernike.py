
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




'''
# Zernike polynomials in cartesian coordinates
def Zernike_cartesien(coefficients, x,y):
 Z = [0]+coefficients
 r = np.sqrt(x**2 + y**2)
 Z1  =  Z[1]  * 1
 Z2  =  Z[2]  * 2*x
 Z3  =  Z[3]  * 2*y

 Z4  =  Z[4]  * np.sqrt(3)*(2*r**2-1)
 Z5  =  Z[5]  * 2*np.sqrt(6)*x*y
 Z6  =  Z[6]  * np.sqrt(6)*(x**2-y**2)
 Z7  =  Z[7]  * np.sqrt(8)*y*(3*r**2-2)
 Z8  =  Z[8]  * np.sqrt(8)*x*(3*r**2-2)
 Z9  =  Z[9]  * np.sqrt(8)*y*(3*x**2-y**2)
 Z10 =  Z[10] * np.sqrt(8)*x*(x**2-3*y**2)
 Z11 =  Z[11] * np.sqrt(5)*(6*r**4-6*r**2+1)
 Z12 =  Z[12] * np.sqrt(10)*(x**2-y**2)*(4*r**2-3)
 Z13 =  Z[13] * 2*np.sqrt(10)*x*y*(4*r**2-3)
 Z14 =  Z[14] * np.sqrt(10)*(r**4-8*x**2*y**2)
 Z15 =  Z[15] * 4*np.sqrt(10)*x*y*(x**2-y**2)
 Z16 =  Z[16] * np.sqrt(12)*x*(10*r**4-12*r**2+3)
 Z17 =  Z[17] * np.sqrt(12)*y*(10*r**4-12*r**2+3)
 Z18 =  Z[18] * np.sqrt(12)*x*(x**2-3*y**2)*(5*r**2-4)
 Z19 =  Z[19] * np.sqrt(12)*y*(3*x**2-y**2)*(5*r**2-4)
 Z20 =  Z[20] * np.sqrt(12)*x*(16*x**4-20*x**2*r**2+5*r**4)
 Z21 =  Z[21] * np.sqrt(12)*y*(16*y**4-20*y**2*r**2+5*r**4)
 Z22 =  Z[22] * np.sqrt(7)*(20*r**6-30*r**4+12*r**2-1)
 Z23 =  Z[23] * 2*np.sqrt(14)*x*y*(15*r**4-20*r**2+6)
 Z24 =  Z[24] * np.sqrt(14)*(x**2-y**2)*(15*r**4-20*r**2+6)
 Z25 =  Z[25] * 4*np.sqrt(14)*x*y*(x**2-y**2)*(6*r**2-5)
 Z26 =  Z[26] * np.sqrt(14)*(8*x**4-8*x**2*r**2+r**4)*(6*r**2-5)
 Z27 =  Z[27] * np.sqrt(14)*x*y*(32*x**4-32*x**2*r**2+6*r**4)
 Z28 =  Z[28] * np.sqrt(14)*(32*x**6-48*x**4*r**2+18*x**2*r**4-r**6)
 Z29 =  Z[29] * 4*y*(35*r**6-60*r**4+30*r**2-4)
 Z30 =  Z[30] * 4*x*(35*r**6-60*r**4+30*r**2-4)
 Z31 =  Z[31] * 4*y*(3*x**2-y**2)*(21*r**4-30*r**2+10)
 Z32 =  Z[32] * 4*x*(x**2-3*y**2)*(21*r**4-30*r**2+10)
 Z33 =  Z[33] * 4*(7*r**2-6)*(4*x**2*y*(x**2-y**2)+y*(r**4-8*x**2*y**2))
 Z34 =  Z[34] * (4*(7*r**2-6)*(x*(r**4-8*x**2*y**2)-4*x*y**2*(x**2-y**2)))
 Z35 =  Z[35] * (8*x**2*y*(3*r**4-16*x**2*y**2)+4*y*(x**2-y**2)*(r**4-16*x**2*y**2))
 Z36 =  Z[36] * (4*x*(x**2-y**2)*(r**4-16*x**2*y**2)-8*x*y**2*(3*r**4-16*x**2*y**2))
 Z37 =  Z[37] * 3*(70*r**8-140*r**6+90*r**4-20*r**2+1)

 ZW = Z1 + Z2 +  Z3
    #+  Z4+  Z5+  Z6+  Z7+  Z8+  Z9+ Z10+ Z11+ Z12+ Z13+ Z14+ Z15+ Z16+ Z17+ Z18+ Z19+Z20+ Z21+ Z22+ Z23+ Z24+ Z25+ Z26+ Z27+ Z28+ Z29+Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
 return Zw
'''

#https://oeis.org/A176988


# Zernike polynomials in polar coordinates
def Zernike_polar(coefficients, r, u, co_num):
 #Z= np.insert(np.array([0,0,0]),3,coefficients)  
 Z =  np.zeros(37)
 Z[:co_num] = coefficients

 #Z1  =  Z[0]  * 1*(np.cos(u)**2+np.sin(u)**2)
 #Z2  =  Z[1]  * 2*r*np.cos(u)
 #Z3  =  Z[2]  * 2*r*np.sin(u)
 
 Z4  =  Z[0]  * np.sqrt(3)*(2*r**2-1)  #defocus

 Z5  =  Z[1]  * np.sqrt(6)*r**2*np.sin(2*u) #astigma
 Z6  =  Z[2]  * np.sqrt(6)*r**2*np.cos(2*u)
 
 Z7  =  Z[3]  * np.sqrt(8)*(3*r**2-2)*r*np.sin(u) #coma
 Z8  =  Z[4]  * np.sqrt(8)*(3*r**2-2)*r*np.cos(u)
 
 Z9  =  Z[5]  * np.sqrt(8)*r**3*np.sin(3*u) #trefoil
 
 Z10=  Z[6] * np.sqrt(8)*r**3*np.cos(3*u)
 
 Z11 =  Z[7] * np.sqrt(5)*(1-6*r**2+6*r**4) #secondary spherical
 
 Z12 =  Z[8] * np.sqrt(10)*(4*r**2-3)*r**2*np.cos(2*u)  #2 astigma
 Z13 =  Z[9] * np.sqrt(10)*(4*r**2-3)*r**2*np.sin(2*u)
 
 Z14 =  Z[10] * np.sqrt(10)*r**4*np.cos(4*u) #tetrafoil
 Z15 =  Z[11] * np.sqrt(10)*r**4*np.sin(4*u)
 
 Z16 =  Z[12] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.cos(u) #secondary coma
 Z17 =  Z[13] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.sin(u)
 
 Z18 =  Z[14] * np.sqrt(12)*(5*r**2-4)*r**3*np.cos(3*u) #secondary trefoil
 Z19 =  Z[15] * np.sqrt(12)*(5*r**2-4)*r**3*np.sin(3*u)
 
 Z20 =  Z[16] * np.sqrt(12)*r**5*np.cos(5*u) #pentafoil
 Z21 =  Z[17] * np.sqrt(12)*r**5*np.sin(5*u)

 Z22 =  Z[18] * np.sqrt(7)*(20*r**6-30*r**4+12*r**2-1) #spherical
 
 Z23 =  Z[19] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.sin(2*u) #astigma
 Z24 =  Z[20] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.cos(2*u)
 
 Z25 =  Z[21] * np.sqrt(14)*(6*r**2-5)*r**4*np.sin(4*u)#trefoil
 Z26 =  Z[22] * np.sqrt(14)*(6*r**2-5)*r**4*np.cos(4*u)
 
 Z27 =  Z[23] * np.sqrt(14)*r**6*np.sin(6*u) #hexafoil 
 Z28 =  Z[24] * np.sqrt(14)*r**6*np.cos(6*u)

 Z29 =  Z[25] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.sin(u) #coma
 Z30 =  Z[26] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.cos(u)
 
 Z31 =  Z[27] * 4*(21*r**4-30*r**2+10)*r**3*np.sin(3*u)#trefoil
 Z32 =  Z[28] * 4*(21*r**4-30*r**2+10)*r**3*np.cos(3*u)

 Z33 =  Z[29] * 4*(7*r**2-6)*r**5*np.sin(5*u) #pentafoil
 Z34 =  Z[30] * 4*(7*r**2-6)*r**5*np.cos(5*u)
 
 Z35 =  Z[31] * 4*r**7*np.sin(7*u) #heptafoil
 Z36 =  Z[32] * 4*r**7*np.cos(7*u)
 
 Z37 =  Z[33] * 3*(70*r**8-140*r**6+90*r**4-20*r**2+1) #spherical
 
#Z1+Z2+Z3+
 ZW = Z4+Z5+Z6+Z7+Z8+Z9+Z10+Z11+Z12+Z13+Z14+Z15+Z16+ Z17+Z18+Z19+Z20+Z21+Z22+Z23+ Z24+Z25+Z26+Z27+Z28+ Z29+ Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
 return ZW

