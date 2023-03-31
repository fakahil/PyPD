
import numpy as np
import matplotlib.pyplot as plt



def mask_pupil(rpupil, size):
 r = 1
 xx = np.linspace(-r, r, 2*rpupil)
 yy = np.linspace(-r, r, 2*rpupil) 
 [X,Y] = np.meshgrid(xx,yy) 
 R = np.sqrt(X**2+Y**2)
 theta = np.arctan2(Y, X)
 M = 1*(np.cos(theta)**2+np.sin(theta)**2)
 M[R>1] = 0
 Mask =  np.zeros([size,size])
 Mask[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1]= M
 return Mask




