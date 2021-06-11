# PyPD

This repo contains a Python package that I have written for applying Phase Diversity Reconstruction on Solar Orbiter data (or any other telescope with a circular aperture). It could be applied on any astronomical data as well. 

It is based on the algorithm described in [Löfdahl and Scharmer. 1994](http://adsabs.harvard.edu/full/1994A&AS..107..243L)

# How to use the code?

## The `Telescope` class:
The parameters needed for this class are:
- `lam`: the wavelength of study
- `diameter`: the diameter of the telescope's aperture 
- `focal_length`: the effective focal length of the telescope
- `name`: the name of your telescope
- `platescale`: the pixel size in arcseconds
- `size`: the size of the detector in pixels

This class allows the user to compute


## The `minimization` class:

This class allows the user to:

1- Minimize the "Error Metric" and get the Zernike Polynomials describing the wavefront error of the telescope
2- Plot the wavefront error, the point spreas function (PSF), the modulation transfer function (MTF), the 1D azimuthal MTF of the telescope 
3- Construct the PSF and MTF from a set of Zernike Polynomials (from wavefront to PSF to OTF to MTF)
4- Restore blurred images and correct for the PSF choosing between the Wiener filter and the Richardson-Lucy filter
5- 
