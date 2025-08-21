## Steps that I've done

Steps for installing found in *ViSP-Inversion/desire-v5.06/doc/installation.txt*.

I created a new directory in the "run" directory and copied "desire.dtrol", "keyword.input" and the "input" folder included in "run/example".

How i think it runs as of now (aug 21):
- cd into the new directory you made in the "run" directory. Mine is called "custom_test"
- then run `../../bin/desire desire.dtrol`
- right now it errors because there is no "observed" file thingy -- not sure how that's formatted yet...

__

Pre-processing is done with the *ViSP_inversion.py* file.

- in the *main* function, change `dataset_root` to the directory where your data is located and `fits_directory` to where you want the pre-processed output fits files to be(?).
- I made `aux_data_dir` a relative path since it's one of the things downloaded in this repo
- I forget what `fiducial_arm_ID = 3` and `fiducial_pol_ID = 1` do...

  ^ *maybe instead of running through han's pre-porcessing, I manually configure the necessary output files from the pre-processing i've already done?*

in *initialization.input*, change things there too

Once the *ViSP_inversion.py* file is configured, make sure the `dkist` conda env is activated and run `python ViSP_inversion` in the terminal.


---

# DKIST Level 2 pipeline for inversion of ViSP data

## 1 Identify invertible data sets:

[Thunderbolts](https://thunderbolts.dev.dkistdc.nso.edu/proposals)


## 2 Pre-processing pipeline and tools

The pre-processing pipeline is contained in the file *ViSP_inversion.py*,
aided by tools in the *ViSP_tools.py* package.

### 2.1 Steps in the pre-processing pipeline include:

+ Specification of data sets to be inverted together, up to three arms.
+ Verification and analysis of the assiciated .asdf files.
+ Alignment of the datasets, scaling based on ViSP hairlines. Data are
   scaled to common spatial scace relative to a specified fiducial arm.
   Shifts between arms, including those due to differential refraction
   are corrected.
+ Wavelength calibration, based on initial guess of the dispersion,
   and position of the deepest line in the spectral range.
+ Determination of cosine of the heliocentric viewing angle, \mu
+ Calibration of intensities based on Atlas, heliocentric angle and
   theoretical center-to-limb behavior.
+ Rebin of data to achieve as close as possible "square pixels",
   determined by number of pixels along the slit relative to the slit width.
+ Adhoc removal of polarization cross-talk.
+ Determination of the spectral broadening by comparison with Atlas
   profiles at the corresponding \mu value.
+ Writing the data, one file per arm, in format for DeSIRe inversions
+ Write auxiliary files for wavelength grid and spectral PSF.

### 2.2 Fits files with theoretical line shape data as function of mu

The *Aux_data* directory holds fits file with calculated line shape profiles 
as a function of 100 mu values. This is used to estimate the pseudo continuum
in cases where the observed wavelength grid does not reach the actual continuum 
e.g., in the case of Ca II 854.2), and to aid in the estimate of the spectral PSF.

### 2.3 Setting up the Python environment

The ViSP preprocessing pipeline relies on tools in the
[dkist tools package](https://docs.dkist.nso.edu/projects/python-tools/en/latest/installation.html),
as supported by the DKIST data center. 

Here is a list of additional packages to install for the ViSP pre-processing pipeline. These instructions
assume you have one of the anaconda variants installed for your package management.

+ conda create -n dkist
+ conda activate dkist
+ conda install -c conda-forge dkist
+ conda install -c conda-forge numba
+ conda install -c conda-forge mda-xdrlib
+ conda install -c conda-forge scikit-learn

You may also want to install the jupyter package, to make plots, run tests.

+ conda install -c conda-forge jupyter

The main set of routines *ViSP_inversion.py* depends on routines in the file ViSP_tools.py.
You will have to make this file visble to python by including the path to it in your
*PYTHONPATH* environment variable:

+ export PYTHONPATH={$HOME}/your/ViSP_package/directory
  
Finally, the code needs to be able to read the solar disk-center atlas as implemented in the RH code.
To indstall you will have to download the sub-directory *python/rhanalyze" from the [RH distribution
on GitHub: ](https://github.com/han-uitenbroek/RH) and add that to your PYTHONPATH environment variable:

+ export PYTHONPATH=${PYTHONPATH}:${HOME}/your/rhanalyze/directory

## 3.0 Inversions with DeSIRe

### 3.1 Installing the DeSIRe inversion code

File *desire-v5.06.tgz* contains the DeSIRe inversion code. To install unpack the tar file and run `make FC=gfortran install`
in the src directory.

### 3.2 Python wrapper for parallel processing in DeSIRe

The file *parallel_master.tgz* contains python code to execute DeSIRe in parrallel. Unpack the tar file and compile the included DeSIRe code.


