import os
import json
import sys

import numpy as np
import healpy as hp

from pathlib import Path
from argparse import ArgumentParser

import bilby
import corner

from bilby_likelihoods import PoissonLikelihood, PoissonRMSLikelihood, MultiPoissonLikelihood
import process_catalog as cat

def estimate_dipole(catalog, priors, injection_parameters, fit_noise, results_dir, flux_cut, snr_cut, clean):
    '''
    Estimate the dipole
    '''
    if catalog.NSIDE is not None:
        obs = catalog.hpx_map[~catalog.hpx_mask]
        pix = catalog.hpx_mask
        mean_counts = np.mean(obs)
    else:
        obs = catalog.pointings['n_sources']
        pix = [catalog.pointings['theta'], catalog.pointings['phi']]
        mean_counts = np.mean(obs)

    N = np.sum(obs)
    mean = np.mean(obs)
    print(f'Total number of sources {N}, mean sources per pixel {np.mean(obs)} in {len(obs)} pixels')

    if fit_noise:
        results_label = f'{catalog.cat_name}_NSIDE{catalog.NSIDE}_powerlaw_snrcut{snr_cut}'

        # Determine median rms and do power law fit
        if catalog.NSIDE is not None:
            rms = catalog.median_healpix_map(catalog.NSIDE)
            rms = rms[~catalog.hpx_mask]
        else:
            rms = catalog.pointings['rms']

        pow_norm, pow_index = catalog.rms_power_law(rms, obs, plot=True)

        priors['rms_amp'] = bilby.core.prior.Uniform(0,2*pow_norm,'$\\mathcal{M}$')
        priors['rms_pow'] = bilby.core.prior.Uniform(0,3,'x')
        injection_parameters['rms_amp'] = pow_norm
        injection_parameters['rms_pow'] = pow_index

        likelihood = PoissonRMSLikelihood(obs, rms, pix, catalog.sigma_ref, catalog.NSIDE)
        result = bilby.run_sampler(likelihood=likelihood,
                                   priors=priors,
                                   sampler='emcee',
                                   nwalkers=15,
                                   ndim=5,
                                   iterations=5000,
                                   injection_parameters=injection_parameters,
                                   outdir=results_dir,
                                   label=results_label,
                                   clean=clean)

    else:
        results_label = f'{catalog.cat_name}_NSIDE{catalog.NSIDE}_fluxcut{flux_cut:.2g}'

        if catalog.NSIDE is not None:
            catalog.smoothed_map(mean_counts)
        catalog.plot_poisson(obs, mean_counts)

        priors['lambda'] = bilby.core.prior.Uniform(0, 2*mean_counts, '$\\lambda$')
        injection_parameters['lambda'] = mean_counts

        likelihood = PoissonLikelihood(obs, pix, catalog.NSIDE)
        result = bilby.run_sampler(likelihood=likelihood,
                                   priors=priors,
                                   sampler='emcee',
                                   nwalkers=10,
                                   ndim=4,
                                   iterations=5000,
                                   injection_parameters=injection_parameters,
                                   outdir=results_dir,
                                   label=results_label,
                                   clean=clean)

    return result

def estimate_multi_dipole(catalogs, priors, injection_parameters, results_dir, flux_cuts, clean):
    '''
    Estimate the dipole for multiple catalogs
    '''
    all_obs = []
    all_pix = []
    cat_names = []
    NSIDES = []
    for i, catalog in enumerate(catalogs):
        cat_obs = catalog.hpx_map[~catalog.hpx_mask]
        cat_pix = catalog.hpx_mask

        N = np.sum(cat_obs)
        print(f'Total number of sources {N} for {catalog.cat_name},',
              f'mean sources per pixel {np.mean(cat_obs)} in {len(cat_obs)} pixels')

        mean_counts = np.mean(cat_obs)
        priors[f'lambda_{i}'] = bilby.core.prior.Uniform(0, 2*mean_counts, f'$\\lambda_{i}$')
        injection_parameters[f'lambda_{i}'] = mean_counts

        all_obs.append(cat_obs)
        all_pix.append(cat_pix)
        cat_names.append(catalog.cat_name)
        NSIDES.append(catalog.NSIDE)

    likelihood = MultiPoissonLikelihood(all_obs, all_pix, NSIDES)
    result = bilby.run_sampler(likelihood=likelihood,
                               priors=priors,
                               sampler='emcee',
                               nwalkers=10,
                               ndim=3+len(catalogs),
                               iterations=10000,
                               injection_parameters=injection_parameters,
                               outdir=results_dir,
                               label=f"{'_'.join(cat_names)}_NSIDE{catalog.NSIDE}",
                               clean=clean)

    return result

def save_results(result, catalogs, flux_cuts, snr_cut, nside, results_dir):
    '''
    Get result and configuration and save in dict
    '''
    current_result = {}

    current_result['parameters'] = {}
    current_result['parameters']['catalogs'] = ','.join([catalog.cat_name for catalog in catalogs])
    current_result['parameters']['nsources'] = int(sum([catalog.total_sources for catalog in catalogs]))
    current_result['parameters']['nside'] = nside
    current_result['parameters']['flux_cut'] = ','.join([str(flux_cut) for flux_cut in flux_cuts])
    current_result['parameters']['snr_cut'] = str(snr_cut)

    current_result['results'] = {}
    keys = result.search_parameter_keys
    for key in keys:
        stats = result.get_one_dimensional_median_and_error_bar(key)
        print(f'{key} = {stats.median:.3g} - {stats.minus:.3g} + {stats.plus:.3g}')
        current_result['results'][key] = stats.median
        current_result['results'][key+'_minus'] = stats.minus
        current_result['results'][key+'_plus'] = stats.plus

    results_json = os.path.join(results_dir,'poisson_results.json')
    if not os.path.exists(results_json):
        results_dict = []
    else:
        with open(results_json, 'r') as infile:
            results_dict = json.load(infile)

    matched = False
    for i, result in enumerate(results_dict):
        if current_result['parameters'] == result['parameters']:
            results_dict[i] = current_result
            matched = True
    if not matched:
        results_dict.append(current_result)

    with open(results_json, 'w') as outfile:
        json.dump(results_dict, outfile, indent=4)

def main():
    parser = new_argument_parser()
    args = parser.parse_args()

    name = args.catalog_name
    nside = args.nside
    flux_cut = args.flux_cut
    snr_cut = args.snr_cut
    results_dir = args.results_dir
    fit_noise = args.fit_noise
    keep_results = args.keep_results

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Set up dipole
    injection_parameters = dict(amp=4.5e-3,
                                dipole_ra=168.,
                                dipole_dec=-7.)

    priors = dict()
    priors['amp'] = bilby.core.prior.Uniform(0,1,'$\\mathcal{D}$')
    priors['dipole_ra'] = bilby.core.prior.Uniform(0,360.,'RA')
    priors['dipole_dec'] = bilby.core.prior.Uniform(-90.,90.,'DEC')

    catalogs = []
    flux_cuts = []

    if name == 'SIM':
        #---------BASIC SIMULATION---------------
        catalog, flux_cut_out, dipole_amp = cat.simulated_data(nside, results_dir, flux_cut, snr_cut, fit_noise)
        catalogs.append(catalog)
        flux_cuts.append(flux_cut_out)

        injection_parameters['amp'] = dipole_amp
    else:
        path = Path(__file__).parent / 'parsets' / name
        with open(path.with_suffix('.json')) as infile:
            param_sets = json.load(infile)

        for param_set in param_sets:
            if flux_cut is None:
                flux_cut = param_set['catalog']['default_flux_cut']
            catalog, flux_cut_out = cat.catalog_data(param_set, nside, flux_cut, snr_cut, fit_noise)
            catalogs.append(catalog)
            flux_cuts.append(flux_cut_out)

    clean = not keep_results
    results = []
    if len(catalogs) == 1:
        for i, catalog in enumerate(catalogs):
            result = estimate_dipole(catalog, priors, injection_parameters, 
                                     fit_noise, results_dir, flux_cuts[i], snr_cut, clean)
            result.plot_corner(titles=False, show_titles=True, title_fmt='.3g',
                               color='navy', truth_color='crimson', smooth=1.5)
            results.append(result)

    if len(catalogs) > 1:
        result = estimate_multi_dipole(catalogs, priors, injection_parameters, 
                                       results_dir, flux_cuts, clean)
        result.plot_corner(titles=False, show_titles=True, title_fmt='.3g',
                           color='navy', truth_color='crimson', smooth=1.5)
        save_results(result, catalogs, flux_cuts, snr_cut, nside, results_dir)
    else:
        result = results[0]
        save_results(result, catalogs, flux_cuts, snr_cut, nside, results_dir)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("catalog_name", type=str,
                        help="""Name of the catalog to fit dipole. Name should
                                either match a corresponding json file in the 
                                parsets directory (withouth the extension) or
                                be SIM, in which case a catalogue will be simulated.""")
    parser.add_argument("--nside", default=32, type=int,
                        help="""NSIDE parameter for HEALPix""")
    parser.add_argument("--flux_cut", default=None, type=float,
                        help="""Lower flux density limit, will overwrite the value
                                present in the parameter json file.""")
    parser.add_argument("--snr_cut", default=None, type=float,
                        help="""Lower S/N limit.""")
    parser.add_argument("--results_dir", default='Results', type=str,
                        help="""Directory where to store results""")
    parser.add_argument('--fit_noise', action='store_true',
                        help="""Try to fit noise and counts with a power law (default=False).""")
    parser.add_argument('--keep_results', action='store_true',
                        help="""Specify if you wish to keep results from the estimator. If specified
                                the script will only run the diagnostic plots (default=False).""")

    return parser

if __name__ == '__main__':
    main()