
import numpy as np




#from . import detector
#from .detector import *

class Telescope(object):

    def __init__(self,lam,diameter,focal_length,name,platescale,size):
     self.lam = lam
     self.diameter = diameter
     self.focal_length = focal_length
     self.name = name
     self.nu_cutoff = self.diameter/self.lam
     self.size = size 
     self.platescale = platescale
     #self.camera = Detector()

    def calc_ps(self, pix_size):
      return 206265*pix_size/self.focal_length
    
    def show_lam(self):
        return self.lam

    def show_diameter(self):
        return self.diameter

    def pupil_size(self):
        pix = self.platescale
        pixrad = pix*np.pi/(180*3600)
        
        deltanu = 1./(self.size*pixrad) 
        rpupil = self.nu_cutoff/(2*deltanu) 
        return np.int(rpupil)

    def spatial_resolution(self):
        return self.diameter/self.lam

'''
def pupil_size(D,lam,pix,size):
    pixrad = pix*np.pi/(180*3600)  # Pixel-size in radians
    nu_cutoff = D/lam      # Cutoff frequency in rad^-1
    deltanu = 1./(size*pixrad)     # Sampling interval in rad^-1
    rpupil = nu_cutoff/(2*deltanu) #pupil size in pixels
    return np.int(rpupil)

def calc_plate_scale(pix_size, focal_length):
    return 206265*pix_size/focal_length
'''


 
