
# PyPD

This repo contains a Python package that I have written for applying Phase Diversity Reconstruction on Solar Orbiter data (or any other telescope with a circular aperture). It could be applied on any astronomical data as well. 

It is based on the algorithm described in [Löfdahl and Scharmer. 1994](http://adsabs.harvard.edu/full/1994A&AS..107..243L)

# Classes explained
An important note here is to use a unified unit for all parameters (nm or m)

## The `Telescope` class:
The parameters needed for this class are:
- `lam`: the wavelength of study
- `diameter`: the diameter of the telescope's aperture 
- `focal_length`: the effective focal length of the telescope
- `name`: the name of your telescope
- `platescale`: the pixel size in arcseconds
- `size`: the size of the detector in pixels

This class allows the user to compute:
1. `calc_ps`: platescale in arcsec/pixel with an additional input of `pix_size` in units the same as the focal length
2. `spatial_resolution`: the spatial resolution (Rayleigh limit)
3. `pupil_size`: the size of the exit pupil of the system 

## The `minimization` class:

This class allows the user to:

1. Minimize the "Error Metric" and get the Zernike Polynomials describing the wavefront error of the telescope
2. Plot the wavefront error, the point spread function (PSF), the modulation transfer function (MTF), the 1D azimuthal MTF of the telescope 
3. Construct the PSF and MTF from a set of Zernike Polynomials (from wavefront to PSF to OTF to MTF)
4. Restore blurred images and correct for the PSF choosing between the Wiener filter and the Richardson-Lucy filter

The parameters of this class:
  1. the pair of focused and defocused images, the input image should have a format of `2xsize_xxsize_y`.
  2. the parameters of the `Telescope` class
  3. `cutoff` frequency for the noise filtering
  4.  `reg`: regularization parameter for the Wiener filter
  5. `ap`: the amount of the apodization rim in pixels
  6. `x1, x2, y1, y2`: the delimeters of the region of interest
  7. `co_num`: number of Zernike polynomials to fit the wavefront
  8. `del_z`: the amount of defocus in lambda

The modules of this class:
  1. `fit`: takes in the method of fitting and returns the best-fit Zernike coefficients
  2. `plot_results`: takes in the returned Zernike coefficients and plots the wavefront error, the 2D MTF, the 1D MTF
  3. `restored_scene`: perform the deconvolution of a blurred image with an input of Zernike Coefficients. The user can use between two filter `Wiener` and `richardsonLucy`

## The `patch_pd` class:
For fitting the wavefront error is sub-regions of the full FOV of the PD dataset. The user can enter the size of the subregion (it has to be quadratic) plus the number of Zernike polynomials to be fit and the names of the output files (one for the wavefront error and one for the 2D MTF).

The class offers the option to run parallel computation by setting the parser -p to True (see below).
# How to use the code?
The `minimization` class returns the best-fit zernike polynomials, a visualisation of the results (wavefront error+MTF), and the restored scene (in this case the focused image of the PD dataset). To get these specific results, type in the shell terminal:
```
python3 minimization.py -i 'path/input.fits' -s 150 -w 617.3e-6 -a 140 -f 4125.3 -p 0.5 -c 0.5 -r 1e-10 -ap 10 -x1 500 -x2 650 -y1 500 -y2 650 -z 10 -del 0.5 -o path/reduced.fits -fl 'Wiener'
```

The specific description of the parsers and input to the class can be found inside [the main code](https://github.com/fakahil/PyPD/blob/master/minimization.py). The values given above are for an example PD dataset taken by the PHI/HRT telescope. You can change the values according to your telescope.

To use the `patch_pd` class:

```
python3 patch_pd.py -i 'path/input.fits' -z 10 -d 265 -ow 'path/output_wf.fits' -om 'path/output_wf.fits -p True

```
The parsers description can be found in the [main code](https://github.com/fakahil/PyPD/blob/master/patch_pd.py)



