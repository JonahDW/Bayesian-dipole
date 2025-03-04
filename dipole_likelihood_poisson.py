import os
import json
import sys

import numpy as np
import healpy as hp

from pathlib import Path
from argparse import ArgumentParser

import bilby
import corner

from bilby_likelihoods import *
import process_catalog as cat
import process_pointings as point

def estimate_dipole(catalog, priors, injection_parameters, extra_fit, fit_col,
                    completeness, results_dir, flux_cut, snr_cut, clean):
    '''
    Estimate the dipole with a single catalogue
    '''
    label = f'{catalog.cat_name}'
    if catalog.NSIDE is not None:
        label += f'_NSIDE{catalog.NSIDE}'
        obs = catalog.hpx_map[~catalog.hpx_mask]
        pix = catalog.hpx_mask

        mean_counts = np.mean(obs)
        catalog.smoothed_map(mean_counts)
    else:
        obs = catalog.pointings['n_sources']
        pix = [catalog.pointings['theta'], catalog.pointings['phi']]
        mean_counts = np.mean(obs)

    N = np.sum(obs)
    mean = np.mean(obs)
    print(f'Total number of sources {N}')
    print(f'Mean sources per pixel {np.mean(obs)} in {len(obs)} pixels')

    if snr_cut is not None:
        label += f'_snrcut{snr_cut:g}'

    if extra_fit == 'noise':
        label += f'_powerlaw'

        # Determine median rms and do power law fit
        if catalog.NSIDE is not None:
            rms = catalog.median_healpix_map(catalog.NSIDE, catalog.rms_col)
            rms = rms[~catalog.hpx_mask]
        else:
            rms = catalog.pointings['rms']

        pow_norm, pow_index = catalog.rms_power_law(rms, obs, plot=True)

        priors['rms_amp'] = bilby.core.prior.Uniform(0,2*pow_norm,
                                                     '$\\mathcal{M}$')
        priors['rms_pow'] = bilby.core.prior.Uniform(0,3,'x')
        injection_parameters['rms_amp'] = pow_norm
        injection_parameters['rms_pow'] = pow_index

        likelihood = PoissonRMSLikelihood(obs, rms, pix, 
                                          catalog.sigma_ref, 
                                          catalog.NSIDE)
        result = bilby.run_sampler(likelihood=likelihood,
                                   priors=priors,
                                   sampler='emcee',
                                   nwalkers=15,
                                   ndim=5,
                                   iterations=5000,
                                   injection_parameters=injection_parameters,
                                   outdir=results_dir,
                                   label=label,
                                   clean=clean)

    elif extra_fit == 'linear':
        label += f'_fluxcut{flux_cut:g}_linear_{fit_col}'
        if completeness is not None:
            label += '_'+completeness

        # Get values to fit linear relation to
        if catalog.NSIDE is not None:
            if fit_col in catalog.hpx_table.colnames:
                fit_vals = catalog.hpx_table[fit_col]
            elif fit_col in catalog.catalog.colnames:
                print(f'Creating median HEALPix map of {fit_col}')
                fit_vals = catalog.median_healpix_map(catalog.NSIDE, fit_col)
            else:
                print(f'Specified fit column, {fit_col}, not found')
                sys.exit()
            fit_vals = fit_vals[~catalog.hpx_mask]
        else:
            fit_vals = catalog.pointings[fit_col]
        max_val = np.max(np.abs(fit_vals))

        priors['lambda'] = bilby.core.prior.Uniform(0, 2*mean_counts, '$\\lambda$')
        priors['lin_amp'] = bilby.core.prior.Uniform(-1/max_val, 1/max_val,'$a$')

        injection_parameters['lambda'] = mean_counts
        injection_parameters['lin_amp'] = 0

        likelihood = PoissonLinearLikelihood(obs, fit_vals, 
                                             pix, catalog.NSIDE)
        result = bilby.run_sampler(likelihood=likelihood,
                                   priors=priors,
                                   sampler='emcee',
                                   nwalkers=15,
                                   ndim=5,
                                   iterations=5000,
                                   injection_parameters=injection_parameters,
                                   outdir=results_dir,
                                   label=label,
                                   clean=clean)

    else:
        label += f'_fluxcut{flux_cut:g}'
        if completeness is not None:
            label += '_'+completeness

        catalog.plot_poisson(obs, mean_counts)

        priors['lambda'] = bilby.core.prior.Uniform(0, 2*mean_counts,
                                                    '$\\lambda$')
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
                                   label=label,
                                   clean=clean)

    return result

def estimate_multi_dipole(catalogs, priors, injection_parameters, 
                          results_dir, dipole_mode, clean):
    '''
    Estimate the dipole for multiple catalogs
    '''
    x         = []
    alpha     = []
    all_obs   = []
    all_pix   = []
    NSIDES    = []
    cat_names = []
    for i, catalog in enumerate(catalogs):
        cat_obs = catalog.hpx_map[~catalog.hpx_mask]
        cat_pix = catalog.hpx_mask

        N = np.sum(cat_obs)
        print(f'Total number of sources {N} for {catalog.cat_name}')
        print(f'Mean sources per pixel {np.mean(cat_obs)} in {len(cat_obs)} pixels')

        mean_counts = np.mean(cat_obs)
        priors[f'lambda_{i}'] = bilby.core.prior.Uniform(0, 2*mean_counts, 
                                                         f'$\\lambda_{i}$')
        injection_parameters[f'lambda_{i}'] = mean_counts

        all_obs.append(cat_obs)
        all_pix.append(cat_pix)
        cat_names.append(catalog.cat_name)
        NSIDES.append(catalog.NSIDE)
        if dipole_mode.lower() != 'amplitude':
            alpha.append(catalog.alpha)
            x.append(catalog.x)

    results_label = f"{'_'.join(cat_names)}_NSIDE{catalog.NSIDE}"
    ndim = 3+len(catalogs)

    if dipole_mode.lower() == 'amplitude':
        likelihood = MultiPoissonLikelihood(all_obs, all_pix, NSIDES)

    if dipole_mode.lower() == 'velocity':
        results_label += '_velocity'
        likelihood = MultiPoissonVelocityLikelihood(all_obs, all_pix, 
                                                    alpha, x, NSIDES)

    if dipole_mode.lower() == 'combined':
        results_label += '_combined'
        ndim += 1
        likelihood = MultiPoissonCombinedLikelihood(all_obs, all_pix, 
                                                    alpha, x, NSIDES)

    if dipole_mode.lower() == 'combined-dir':
        results_label += '_combined-dir'
        ndim += 3
        likelihood = MultiPoissonCombinedDirLikelihood(all_obs, all_pix,
                                                       alpha, x, NSIDES)

    result = bilby.run_sampler(likelihood=likelihood,
                               priors=priors,
                               sampler='emcee',
                               nwalkers=5*ndim,
                               ndim=ndim,
                               iterations=5000,
                               injection_parameters=injection_parameters,
                               outdir=results_dir,
                               label=results_label,
                               clean=clean)

    return result

def save_results(result, catalogs, flux_cuts, snr_cut, nside, results_dir,
                 extra_fit=None, fit_col=None, completeness=None):
    '''
    Get result and configuration and save in dict
    '''
    current_result = {}

    current_result['parameters'] = {}
    current_result['parameters']['catalogs'] = ','.join([catalog.cat_name for catalog in catalogs])
    current_result['parameters']['nsources'] = int(sum([catalog.total_sources for catalog in catalogs]))
    current_result['parameters']['nside'] = nside
    current_result['parameters']['flux_cut'] = ','.join([str(flux_cut) for flux_cut in flux_cuts])
    if snr_cut is not None:
        current_result['parameters']['snr_cut'] = str(snr_cut)
    if extra_fit is not None:
        current_result['parameters']['extra_fit'] = extra_fit
    if fit_col is not None:
        current_result['parameters']['fit_col'] = fit_col
    if completeness is not None:
        current_result['parameters']['completeness'] = completeness

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
    extra_fit = args.extra_fit
    fit_col = args.fit_col
    completeness = args.completeness
    dipole_mode = args.dipole_mode
    keep_results = args.keep_results

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Set up dipole parameters
    priors = dict()
    injection_parameters = dict()
    if dipole_mode.lower() == 'amplitude':
        priors['amp'] = bilby.core.prior.Uniform(0,1,'$\\mathcal{D}$')
        injection_parameters['amp'] = 4.5e-3
    if dipole_mode.lower() == 'velocity':
        priors['beta'] = bilby.core.prior.Uniform(0,0.1,'$\\beta$')
        injection_parameters['beta'] = 1.23e-3
    if dipole_mode.lower() == 'combined':
        priors['amp'] = bilby.core.prior.Uniform(0,0.5,'$\\mathcal{D}$')
        priors['beta'] = bilby.core.prior.Uniform(0,0.05,'$\\beta$')
        injection_parameters['amp'] = 4.5e-3
        injection_parameters['beta'] = 1.23e-3
    if dipole_mode.lower() == 'combined-dir':
        priors['amp'] = bilby.core.prior.Uniform(0,0.5,'$\\mathcal{D}$')
        priors['beta'] = 1.23e-3 #bilby.core.prior.Uniform(0,0.05,'$\\beta$')
        priors['vel_ra'] = 168. #bilby.core.prior.Uniform(0,360.,'vel RA')
        priors['vel_dec'] = -7. #CosineDeg(-90.,90.,'vel DEC')

        injection_parameters['amp'] = 4.5e-3
        injection_parameters['beta'] = 1.23e-3
        injection_parameters['vel_ra'] = 168.
        injection_parameters['vel_dec'] = -7.

    priors['dipole_ra'] = bilby.core.prior.Uniform(0,360.,'RA')
    priors['dipole_dec'] = CosineDeg(-90.,90.,'DEC')

    injection_parameters['dipole_ra'] = 140.
    injection_parameters['dipole_dec'] = -7.

    catalogs = []
    flux_cuts = []

    path = Path(__file__).parent / 'parsets' / name
    with open(path.with_suffix('.json')) as infile:
        param_sets = json.load(infile)

    for param_set in param_sets:
        if flux_cut is None:
            flux_cut_out = param_set['catalog']['default_flux_cut']
        else:
            flux_cut_out = flux_cut

        # Check if catalogue has pointings
        if 'pointing_col' in param_set['columns']:
            catalog, flux_cut_out = point.catalog_data(param_set, 
                                                       flux_cut_out,
                                                       snr_cut, 
                                                       extra_fit,
                                                       completeness)
        # Otherwise it is a regular catalogue
        else:
            catalog, flux_cut_out = cat.catalog_data(param_set, nside,
                                                     flux_cut_out,
                                                     snr_cut, extra_fit)

        if dipole_mode.lower() != 'amplitude':
            catalog.alpha = param_set['catalog']['alpha']
            catalog.x = param_set['catalog']['x']
        catalogs.append(catalog)
        flux_cuts.append(flux_cut_out)

    clean = not keep_results
    results = []
    if len(catalogs) == 1:
        for i, catalog in enumerate(catalogs):
            result = estimate_dipole(catalog, priors, injection_parameters, 
                                     extra_fit, fit_col, completeness,
                                     results_dir, flux_cuts[i], snr_cut, clean)
            result.plot_corner(titles=False, show_titles=True, title_fmt='.3g',
                               color='navy', truth_color='crimson', smooth=1.5)
            results.append(result)

    if len(catalogs) > 1:
        result = estimate_multi_dipole(catalogs, priors, injection_parameters, 
                                       results_dir, dipole_mode, clean)
        result.plot_corner(titles=False, show_titles=True, title_fmt='.3g',
                           color='navy', truth_color='crimson', smooth=1.5)
        save_results(result, catalogs, flux_cuts, snr_cut, nside, results_dir)
    else:
        result = results[0]
        save_results(result, catalogs, flux_cuts, snr_cut, nside, results_dir,
                     extra_fit, fit_col, completeness)

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
    parser.add_argument('--extra_fit', default=None, type=str,
                        help="""Fit additional relation to the data to model 
                                systematic effects. Current options are 
                                'noise' and 'linear' (default = no additional fit)""")
    parser.add_argument('--fit_col', default=None, type=str,
                        help="""Which column in the catalogue to use for extra fit.""")
    parser.add_argument('--completeness', nargs='?', const=True,
                        help="""Use completeness when looking at number counts, provided a 
                                column which stores the completeness of each source in the 
                                catalog (default = do not use completeness).""")
    parser.add_argument('--dipole_mode', default='amplitude', type=str,
                        help="""How to break down the dipole amplitude when fitting 
                                multiple catalogues. Default mode 'amplitude' fits 
                                for amplitude, 'velocity' fits velocity directly. 
                                'Combined' fits both, assuming separate velocity 
                                component and intrinsic dipole. Latter options 
                                require specification of spectral index and power 
                                law index of flux distribution (x) in json file""")
    parser.add_argument('--keep_results', action='store_true',
                        help="""Specify if you wish to keep results from the estimator.
                                If specified the script will only run the diagnostic 
                                plots (default=False).""")

    return parser

if __name__ == '__main__':
    main()
