<div align="center">

  # Bayesian-dipole 
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7962922.svg)](https://doi.org/10.5281/zenodo.7962922)

</div>

The purpose of this module is to estimate parameters of the cosmic number count dipole from catalogues of (radio) sources using Bayesian methods. Catalogues are pixelised using the python implementation of HEALPix, [healpy](https://healpy.readthedocs.io/en/latest/). The software used for Bayesian inference is [bilby](https://lscsoft.docs.ligo.org/bilby/installation.html), which can wrap around a range of different samplers. These scripts use [emcee](https://emcee.readthedocs.io/en/stable/) for sampling.

## dipole_likelihood.py

Script to be run to perform inference on a given catalogue or set of catalogues. The setup for any given catalogue (or set of catalogues) is defined in a json file in the `parsets` directory, several examples are already present. The name of the json file (without the extension) is given as input. This json file must contain the details about the catalogue(s), like the filename and column names, as well as any cuts in the data. Example usage:

```python dipole_likelihood.py NVSS --nside 32 --flux_cut 15```

Which will read the catalogue details from `parsets/NVSS.json`, pixelises the catalogue using HEALPix with NSIDE=32, and cuts out all sources below a flux density of 15 mJy. Other more general catalogue cuts should be specified in the json file.

If the json file contains multiple catalogues, the multi-catalogue estimator will be used. In this case the `dipole_mode` argument can be used to determine what kind of dipole model is assumed. In this case, if any velocity component is fit, the determined alpha and x will need to be included in the parameter file. For an example of this, see `parsets/NVSS-RACS-WISE.json`.

If the json dictionary contains the `pointing_columns` key, the catalogue will be treated as containing a series of pointings and will not be discretised using HEALPix. For an example of this see `parsets/MALS.json`.

### Full list of arguments
```
usage: dipole_likelihood.py [-h] [--nside NSIDE] [--flux_cut FLUX_CUT]
                            [--results_dir RESULTS_DIR] [--snr_cut SNR_CUT]
                            [--extra_fit EXTRA_FIT] [--fit_val FIT_VAL]
                            [--healpix_mask HEALPIX_MASK] [--invert_mask]
                            [--completeness [COMPLETENESS]]
                            [--dipole_mode DIPOLE_MODE]
                            [--statistic STATISTIC] [--keep_results]
                            catalog_name

positional arguments:
  catalog_name          Name of the catalog to fit dipole. Name should either
                        match a corresponding json file in the parsets
                        directory (withouth the extension) or a HEALPix map,
                        with a .fits extension.

optional arguments:
  -h, --help            show this help message and exit
  --nside NSIDE         nside parameter for HEALPix
  --flux_cut FLUX_CUT   Lower flux density limit, will overwrite the value
                        present in the parameter json file.
  --results_dir RESULTS_DIR
                        Directory where to store results
  --snr_cut SNR_CUT     Lower S/N limit.
  --extra_fit EXTRA_FIT
                        Fit additional relation to the data to model
                        systematic effects. Current options are 'noise' and
                        'linear' (default = no additional fit)
  --fit_val FIT_VAL     Which column in the catalogue to use for extra fit.
                        Alternatively, a HEALPix map can be specified, which
                        should end in '.fits'.
  --healpix_mask HEALPIX_MASK
                        Additional optional HEALPix mask.
  --invert_mask         Invert HEALPix mask specified in --healpix_mask.
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
  --statistic STATISTIC
                        Use 'poisson' or 'nb' for negative binomial
  --keep_results        Specify if you wish to keep results from the
                        estimator. If specified the script will only run the
                        diagnostic plots (default=False).
```
## simulate_observations.py

Create a simulated catalog. A catalogue with sources in pointings can be created by using the argument `pointings`, and a contiguous sky catalogue can be created using the the argument `sky`. The catalog parameters, including the injected dipole, can be adjusted in either `parsets/sim-pointings.json` or `parsets/sim-sky.json`. In case of pointings simulation, a catalogue of pointings must be specified which includes R.A., Dec., name and rms of each pointing. For a sky simulation, rms can be constant, or a specified HEALPix rms map as input for local rms. 

```
usage: simulate_observations.py [-h] [--debug] catalog_type

positional arguments:
  catalog_type  Type of catalogue to produce, can be either 'sky', generating
                a contiguous sky survey, or 'pointing', generating a pointing
                based catalogue.

optional arguments:
  -h, --help    show this help message and exit
  --debug       Display debug messages or plots.
```

## systematics_check.py

Check a catalogue for systematics by plotting and computing correlation statistics between key observables and observed numbers counts, both with and without catalogue cuts. As with the likelihood script, simply specify the name of the corresponding json file in the `parsets` directory.

```
usage: systematics_check.py [-h] [--nside NSIDE] [--flux_cut FLUX_CUT]
                            [--columns COLUMNS [COLUMNS ...]]
                            [--haslam_map HASLAM_MAP]
                            catalog_name

positional arguments:
  catalog_name          Name of the catalog to fit dipole. Name should match a
                        corresponding json file in the parsets directory
                        (withouth the extension).

optional arguments:
  -h, --help            show this help message and exit
  --nside NSIDE         NSIDE parameter for HEALPix map
  --flux_cut FLUX_CUT   Lower flux density limit, will overwrite the value
                        present in the parameter json file.
  --columns COLUMNS [COLUMNS ...]
                        Extra columns to fit for correlations.
  --haslam_map HASLAM_MAP
                        Path to Haslam map to correlate with data.
```

## Citation, relevant papers and versions

If you use this code and wish to cite it, you can do so as follows:

```
Wagenveld, J.D. 2023, JonahDW/Bayesian-dipole: Bayesian dipole inference, Zenodo, DOI: 10.5281/zenodo.7962922
```

Depending on the functionality/version used, relevant works can be cited as specified below:

### Current version (03.2026) 

Included functionality to include Negative Binomial statistics in addition to the default Poisson statistics. The use of Negative Binomial in dipole estimation is described in [Böhme et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025PhRvL.135t1001B/abstract).

### v0.3 (03.2025) 

Included functionality to include different contributions (kinematic or instrinsic) contributions to the overall dipole amplitude when multiple catalogues are used. This was used in [Wagenveld et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025A%26A...697A.112W/abstract) on the NVSS, RACS-low, and CatWISE catalogues to perform joint dipole estimations to disentangle kinematic and residual contributions to the total dipole amplitude.

### v0.2 (08.2024)

Included functionality to handle 'pointings', meaning discrete directions on sky. This can be used instead of discretising a contiguous survey with HEALPix. In [Wagenveld et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240816619W/abstract), this was used on the [MALS DR2](https://mals.iucaa.in/releases) catalogue. 

### v0.1 (05.2023)

First release, including a standard Poisson estimator, as well as the rms estimator and the multi-catalogue estimator. This was used to produce the results in [Wagenveld et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...675A..72W/abstract), in which the NVSS and RACS-low catalogues were used. 
