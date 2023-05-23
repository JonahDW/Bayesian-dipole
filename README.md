# Bayesian-dipole [![DOI](https://zenodo.org/badge/644438254.svg)](https://zenodo.org/badge/latestdoi/644438254)

The purpose of this module is to estimate parameters of the cosmic number count dipole from catalogues of (radio) sources using Bayesian methods. Catalogues are pixelised using the python implementation of HEALPix, [healpy](https://healpy.readthedocs.io/en/latest/). The software used for Bayesian inference is [bilby](https://lscsoft.docs.ligo.org/bilby/installation.html), which can wrap around a range of different samplers. These scripts use [emcee](https://emcee.readthedocs.io/en/stable/) for sampling.

Setup for any given catalogue (or set of catalogues!) is defined in a json file in the `parsets` directory, several examples are already present. The name of the json file (without the extension) is then given to the program. Alternatively, the program can be called with the argument `SIM`, in which case a catalogue is simulated using the parameters specified in `parsets/simulation.json`.

If you use this code and wish to cite it, you can do so as follows:

```
Wagenveld, J.D. 2023, JonahDW/Bayesian-dipole: Bayesian dipole inference v0.1, 
Zenodo, DOI: 10.5281/zenodo.7962923
```

## dipole_likelihood_poisson.py

This is the only script that has to be run to perform inference on a given catalogue. Example usage:

```python dipole_likelihood_poisson.py NVSS --nside 32 --flux_cut 15```

Which will read the catalogue details from `parsets/NVSS.json`, pixelises the catalogue using HEALPix with NSIDE=32, and cuts out all sources below a flux density of 15 mJy. Other more general catalogue cuts should be specified in the json file. 

### Full list of arguments
```
usage: dipole_likelihood_poisson.py [-h] [--nside NSIDE] [--flux_cut FLUX_CUT]
                                    [--snr_cut SNR_CUT]
                                    [--results_dir RESULTS_DIR] [--fit_noise]
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
  --fit_noise           Try to fit noise and counts with a power law
                        (default=False).
  --keep_results        Specify if you wish to keep results from the
                        estimator. If specified the script will only run the
                        diagnostic plots (default=False).
```
