from setuptools import setup, find_packages
setup(name='PyPD',
      version='0.2.1',
      keywords = ('optics', 'telescope', 'zernike','phase diversity'),
      description='Module for Phase diversity analysis',
      license = 'MPS License',
      install_requires = ['numpy>=1.9.3','matplotlib>=1.4.3','unwrap'],
      author='Fatima Kahil',
      author_email='kahil@mps.mpg.de',
      url= 'https://github.com/fakahil/PyPD',
      packages = setuptools.find_packages(),
       classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",],)
