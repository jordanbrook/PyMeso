# PyMeso

An implementation of the linear least squares derivative method for calculating azimuthal shear from Doppler velocity data. This is used operationally for detecting mesocyclones and is based on the work by Miller et al., (2013). 
Additionally, a method for simulating rankine vortices has been provided and was used in the verification process for the LLSD method.

### References
- LLSD Azimuthal Shear Technique ([Miller et al. 2013](https://doi.org/10.1175/WAF-D-12-00065.1))

### Dependencies
- [Py-ART](https://github.com/ARM-DOE/pyart/)
- numpy
- scipy
- numba

### Install
To install PyMeso, you can either download and unpack the zip file of the source code or use git to checkout the repository:

`git clone git@github.com:jordanbrook/PyMeso.git`

To install in your home directory, use:

`python setup.py install --user`

### Use
`notebook/PyMesoNotebook.ipynb` provides examples of LLSD retrieval and vortex simulation
