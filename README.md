# PyPD

This rep contains a Python package that I have written for applying Phase Diversity Reconstruction on Solar Orbiter data. It could be applied on any astronomical data as well. 

It is based on the algorithm described in [Löfdahl and Scharmer. 1994](http://adsabs.harvard.edu/full/1994A&AS..107..243L)

# How to use the code?

## The `Telescope` class:


## The `minimization` class:

This class allows the user to:

- Minimize the "Error Metric" and get the Zernike Polynomials describing the wavefront error of the telescope
- Plot the wavefront error, the point spreas function (PSF), the modulation transfer function (MTF), the 1D azimuthal MTF of the telescope 
- Restore blurred images with the retrieved Zernike coefficients (or any) choosing between the Wiener filter and the Richardson-Lucy filter
- 
