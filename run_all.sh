#! /bin/bash

python aperture.py
python noise.py
python telescope.py
python tools.py
python wavefront.py
python zernike.py
python PD.py
wait 
python main.py
