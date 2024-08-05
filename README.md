# Bayesian-dipole [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7962922.svg)](https://doi.org/10.5281/zenodo.7962922)

The purpose of this module is to estimate parameters of the cosmic number count dipole from catalogues of (radio) sources using Bayesian methods. Catalogues are pixelised using the python implementation of HEALPix, [healpy](https://healpy.readthedocs.io/en/latest/). The software used for Bayesian inference is [bilby](https://lscsoft.docs.ligo.org/bilby/installation.html), which can wrap around a range of different samplers. These scripts use [emcee](https://emcee.readthedocs.io/en/stable/) for sampling.

Setup for any given catalogue (or set of catalogues!) is defined in a json file in the `parsets` directory, several examples are already present. The name of the json file (without the extension) is then given to the program. Alternatively, the program can be called with the argument `SIM`, in which case a catalogue is simulated using the parameters specified in `parsets/simulation.json`.

If you use this code and wish to cite it, you can do so as follows:

```
Wagenveld, J.D. 2023, JonahDW/Bayesian-dipole: Bayesian dipole inference, Zenodo, DOI: 10.5281/zenodo.7962922
```

This code has been used to produce the results in [Wagenveld et al. (2023)](https://arxiv.org/abs/2305.15335). The catalogues that were used are the [NVSS](https://www.cv.nrao.edu/nvss/) and [RACS](https://research.csiro.au/casda/the-rapid-askap-continuum-survey-stokes-i-source-catalogue-data-release-1/) catalogues. 

## dipole_likelihood_poisson.py

This is the only script that has to be run to perform inference on a given catalogue. The catalogue name(s) given must correspond to a json file in the `parsets` directory. This json file must contain the details about the catalogue(s), like the filename and column names, as well as any cuts in the data. Example usage:

```python dipole_likelihood_poisson.py NVSS --nside 32 --flux_cut 15```

Which will read the catalogue details from `parsets/NVSS.json`, pixelises the catalogue using HEALPix with NSIDE=32, and cuts out all sources below a flux density of 15 mJy. Other more general catalogue cuts should be specified in the json file.

If the json file contains multiple catalogues, the multi-catalogue estimator will be used. For an example of this see `parsets/NVSS-RACS.json`. If the json dictionary contains the `pointing_columns` key, the catalogue will be treated as containing a series of pointings and will not be discretised using HEALPix. For an example of this see `parsets/MALS.json`.

### Full list of arguments
```
usage: dipole_likelihood_poisson.py [-h] [--nside NSIDE] [--flux_cut FLUX_CUT]
                                    [--snr_cut SNR_CUT]
                                    [--results_dir RESULTS_DIR]
                                    [--extra_fit EXTRA_FIT]
                                    [--fit_col FIT_COL]
                                    [--completeness [COMPLETENESS]]
                                    [--dipole_mode DIPOLE_MODE]
                                    [--keep_results]
                                    catalog_name

positional arguments:
  catalog_name          Name of the catalog to fit dipole. Name should either
                        match a corresponding json file in the parsets
                        directory (withouth the extension) or be SIM, in which
                        case a catalogue will be simulated.

optional arguments:
  -h, --help            show this help message and exit
  --nside NSIDE         NSIDE parameter for HEALPix
  --flux_cut FLUX_CUT   Lower flux density limit, will overwrite the value
                        present in the parameter json file.
  --snr_cut SNR_CUT     Lower S/N limit.
  --results_dir RESULTS_DIR
                        Directory where to store results
  --extra_fit EXTRA_FIT
                        Fit additional relation to the data to model
                        systematic effects. Current options are 'noise' and
                        'linear' (default = no additional fit)
  --fit_col FIT_COL     Which column in the catalogue to use for extra fit.
  --completeness [COMPLETENESS]
                        Use completeness when looking at number counts,
                        provided a column which stores the completeness of
                        each source in the catalog (default = do not use
                        completeness).
  --dipole_mode DIPOLE_MODE
                        How to break down the dipole amplitude when fitting
                        multiple catalogues. Default mode 'amplitude' fits for
                        amplitude, 'velocity' fits velocity directly.
                        'Combined' fits both, assuming separate velocity
                        component and intrinsic dipole. Latter options require
                        specification of spectral index and power law index of
                        flux distribution (x) in json file
  --keep_results        Specify if you wish to keep results from the
                        estimator. If specified the script will only run the
                        diagnostic plots (default=False).
```
## simulate_observations.py

Create a simulated catalog. A catalogue with sources in pointings can be created by using the argument `pointings`, and a contiguous sky catalogue can be created using the the argument `sky`. The catalog parameters, including the injected dipole, can be adjusted in either `parsets/sim-pointings.json` or `parsets/sim-sky.json`. In case of pointings simulation, a catalogue of pointings must be specified which includes R.A., Dec., name and rms of each pointing. For a sky simulation, rms is currently constant but a HEALPix rms map will be available as input for local rms. Example usage:

```python simulate_observations.py pointing```
