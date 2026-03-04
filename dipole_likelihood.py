import os
import json
import sys

import numpy as np
import healpy as hp

from pathlib import Path
from argparse import ArgumentParser

import bilby
import corner

from utils import likelihoods
from utils import priors as dpriors
from utils import process_catalog as cat
from utils import process_pointings as point

def estimate_dipole(data, priors, injection_parameters, extra_fit, fit_val,
                    statistic, base_label, results_dir, clean):
    """
    Estimate the dipole with a single catalogue
    """
    label = base_label
    ndim = 4

    if data.nside is not None:
        obs = data.hpx_map[~data.hpx_mask]
        pix = data.hpx_mask

        mean_counts = np.mean(obs)
        data.smoothed_map(mean_counts)
    else:
        obs = data.pointings['n_sources']
        pix = [data.pointings['theta'], data.pointings['phi']]
        mean_counts = np.mean(obs)

    N = np.sum(obs)
    print(f'Total number of sources {N}')
    print(f'Mean sources per pixel {mean_counts} in {len(obs)} pixels')

    priors['monopole'] = bilby.core.prior.Uniform(0, 2*mean_counts,'$\\mathcal{M}$')
    injection_parameters['monopole'] = mean_counts
    if statistic == 'nb':
        ndim += 1
        priors['p'] = bilby.core.prior.Uniform(0, 1, 'p')
        injection_parameters['p'] = 0.5

    if extra_fit == 'noise':
        label += f'_powerlaw'

        # Determine median rms and do power law fit
        if data.nside is not None:
            # Check if input is HEALPix map
            if fit_val.endswith('.fits'):
                rms_map = hp.read_map(fit_val)
            else:
                rms_map = data.pix_table['rms']
            rms = rms_map[~data.hpx_mask]

            data.sigma_ref = np.nanmedian(rms[~np.isnan(rms)])
            print(f'Median RMS value {data.sigma_ref} mJy/beam')
        else:
            rms = data.pointings['rms']

        pow_norm, pow_index = data.rms_power_law(rms, obs, plot=True)

        priors['monopole'] = bilby.core.prior.Uniform(0,2*pow_norm,'$\\mathcal{M}$')
        priors['x'] = bilby.core.prior.Uniform(0,3,'x')
        injection_parameters['monopole'] = pow_norm
        injection_parameters['x'] = pow_index

        ndim += 1
        likelihood = likelihoods.RMSLikelihood(obs, rms, pix, 
                                               data.sigma_ref, 
                                               statistic, data.nside)

    elif extra_fit == 'linear':
        label += f'_linear_{fit_val}'

        # Get values to fit linear relation to
        if data.nside is not None:
            fit_vals = data.pix_table[fit_val]
            fit_vals = fit_vals[~data.hpx_mask]
        else:
            fit_vals = data.pointings[fit_val]
        max_val = np.max(np.abs(fit_vals))

        priors['lin_amp'] = bilby.core.prior.Uniform(-1/max_val, 1/max_val,'$a$')
        injection_parameters['lin_amp'] = 0

        ndim += 1
        likelihood = likelihoods.LinearLikelihood(obs, fit_vals, pix, 
                                                  statistic, data.nside)

    elif extra_fit == 'dipole':
        label += f'_doubledipole'

        priors['amp2'] = bilby.core.prior.Uniform(0,1,'$\\mathcal{D}_{extra}$')
        priors['dipole2_ra'] = bilby.core.prior.Uniform(0,360.,'$RA_{extra}$')
        priors['dipole2_dec'] = dpriors.CosineDeg(-90.,90.,'$DEC_{extra}$')

        injection_parameters['amp2'] = 4.5e-4
        injection_parameters['dipole2_ra'] = 0.0
        injection_parameters['dipole2_dec'] = 0.0

        ndim += 3
        likelihood = likelihoods.DoubleDipoleLikelihood(obs, pix, statistic, data.nside)

    else:
        data.plot_poisson(obs, mean_counts)
        likelihood = likelihoods.DipoleLikelihood(obs, pix, statistic, data.nside)

    result = bilby.run_sampler(likelihood=likelihood,
                               priors=priors,
                               sampler='emcee',
                               nwalkers=5*ndim,
                               ndim=ndim,
                               iterations=5000,
                               injection_parameters=injection_parameters,
                               outdir=results_dir,
                               label=label,
                               clean=clean)

    return result

def estimate_multi_dipole(catalogs, priors, injection_parameters, 
                          statistic, results_dir, dipole_mode, clean):
    """
    Estimate the dipole for multiple catalogs
    """
    all_obs, all_pix, nsides, x, alpha, cat_names = [], [], [], [], [], []
    for i, catalog in enumerate(catalogs):
        cat_obs = catalog.hpx_map[~catalog.hpx_mask]
        cat_pix = catalog.hpx_mask

        N = np.sum(cat_obs)
        mean_counts = np.mean(cat_obs)
        print(f'Total number of sources {N} for {catalog.name}')
        print(f'Mean sources per pixel {mean_counts} in {len(cat_obs)} pixels')

        priors[f'monopole_{i}'] = bilby.core.prior.Uniform(0, 2*mean_counts, 
                                                         f'$\\mathcal{{M}}_{i}$')
        injection_parameters[f'monopole_{i}'] = mean_counts

        if statistic == 'nb':
            priors[f'p_{i}'] = bilby.core.prior.Uniform(0, 1, f'p_{i}')
            injection_parameters[f'p_{i}'] = 0.5

        all_obs.append(cat_obs)
        all_pix.append(cat_pix)
        cat_names.append(catalog.name)
        nsides.append(catalog.nside)
        if dipole_mode.lower() != 'amplitude':
            alpha.append(catalog.alpha)
            x.append(catalog.x)

    results_label = f"{'_'.join(cat_names)}_nside{catalog.nside}"
    ndim = 3+len(catalogs)

    if dipole_mode.lower() == 'amplitude':
        likelihood = likelihoods.MultiLikelihood(all_obs, all_pix, statistic, nsides)

    if dipole_mode.lower() == 'velocity':
        results_label += '_velocity'
        likelihood = likelihoods.MultiVelocityLikelihood(all_obs, all_pix, 
                                                         alpha, x, statistic, nside)

    if dipole_mode.lower() == 'combined':
        results_label += '_combined'
        ndim += 1
        likelihood = likelihoods.MultiCombinedLikelihood(all_obs, all_pix, 
                                                         alpha, x, statistic, nsides)

    if dipole_mode.lower() == 'combined-dir':
        results_label += '_combined-dir'
        ndim += 3
        likelihood = likelihoods.MultiCombinedDirLikelihood(all_obs, all_pix,
                                                            alpha, x, statistic, nsides)

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

def save_results(result, catalogs, flux_cuts, nside, results_dir, **kwargs):
    '''
    Get result and configuration and save in dict
    '''
    current_result = {}

    current_result['parameters'] = {}
    current_result['parameters']['catalogs'] = ','.join([catalog.name for catalog in catalogs])
    current_result['parameters']['nsources'] = int(sum([catalog.total_sources for catalog in catalogs]))
    current_result['parameters']['nside'] = nside
    current_result['parameters']['flux_cut'] = ','.join([str(flux_cut) for flux_cut in flux_cuts])
    for key, value in kwargs.items():
        if value is not None:
            current_result['parameters'][key] = value

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

    # Catalog options
    flux_cut = args.flux_cut
    snr_cut = args.snr_cut
    completeness = args.completeness

    # Fitting options
    extra_fit = args.extra_fit
    fit_val = args.fit_val
    dipole_mode = args.dipole_mode
    stat = args.statistic

    healpix_mask = args.healpix_mask
    invert_mask = args.invert_mask
    results_dir = args.results_dir
    keep_results = args.keep_results

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    priors, injection_parameters = dpriors.default_dipole_priors(dipole_mode)

    # If name is fits file, assume HEALPix map
    if name.endswith('.fits'):
        hpx_map, hdr = hp.read_map(name, h=True)
        map_name = os.path.basename(name).rsplit('.',1)[0]
        print(f'--- HEALPix map {map_name} loaded ---')

        nside = dict(hdr)['NSIDE']
        flux_cuts = [None]
        labels = [map_name]

        data = cat.HEALPixData(hpx_map, nside, name=map_name)
        catalogs = [data]

        if healpix_mask is not None:
            data.apply_mask(healpix_mask, invert_mask)

    # Otherwise treat input name as json file
    else:
        path = Path(__file__).parent / 'parsets' / name
        with open(path.with_suffix('.json')) as infile:
            param_sets = json.load(infile)

        catalogs = []
        flux_cuts = []
        labels = []
        for param_set in param_sets:
            if flux_cut is None:
                flux_cut_out = param_set['catalog']['default_flux_cut']
            else:
                flux_cut_out = flux_cut

            # Check if catalogue has pointings
            if 'pointing_col' in param_set['columns']:
                data, flux_cut_out = point.catalog_data(param_set, 
                                                        flux_cut_out,
                                                        snr_cut, 
                                                        extra_fit,
                                                        fit_val,
                                                        completeness)
            # Otherwise it is a regular catalogue
            else:
                data, flux_cut_out = cat.catalog_data(param_set, nside,
                                                      flux_cut_out,
                                                      snr_cut, extra_fit)

            # Create catalog base label
            base_label = f'{data.name}'
            if data.nside is not None:
                base_label += f'_nside{data.nside}'
            if flux_cut_out is not None:
               base_label += f'_fluxcut{flux_cut_out:g}'
            if snr_cut is not None:
                base_label += f'_snrcut{snr_cut:g}'
            if completeness is not None:
                base_label += '_'+completeness

            if dipole_mode.lower() != 'amplitude':
                data.alpha = param_set['catalog']['alpha']
                data.x = param_set['catalog']['x']

            catalogs.append(data)
            flux_cuts.append(flux_cut_out)
            labels.append(base_label)

    clean = not keep_results
    if len(catalogs) == 1:
        result = estimate_dipole(catalogs[0], priors, injection_parameters, 
                                 extra_fit, fit_val, stat, labels[0], 
                                 results_dir, clean)
        result.plot_corner(titles=False, show_titles=True, title_fmt='.3g',
                           color='navy', truth_color='crimson', smooth=1.5)
        save_results(result, catalogs, flux_cuts, nside, results_dir,
                     snr_cut=snr_cut, extra_fit=extra_fit, fit_val=fit_val, 
                     completeness=completeness)

    elif len(catalogs) > 1:
        result = estimate_multi_dipole(catalogs, priors, injection_parameters, 
                                       stat,results_dir, dipole_mode, clean)
        result.plot_corner(titles=False, show_titles=True, title_fmt='.3g',
                           color='navy', truth_color='crimson', smooth=1.5)
        save_results(result, catalogs, flux_cuts, nside, results_dir, 
                     dipole_mode=dipole_mode)


def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("catalog_name", type=str,
                        help="""Name of the catalog to fit dipole. Name should
                                either match a corresponding json file in the 
                                parsets directory (withouth the extension) or
                                a HEALPix map, with a .fits extension.""")
    parser.add_argument("--nside", default=64, type=int,
                        help="""nside parameter for HEALPix""")
    parser.add_argument("--flux_cut", default=None, type=float,
                        help="""Lower flux density limit, will overwrite the value
                                present in the parameter json file.""")
    parser.add_argument("--results_dir", default='Results', type=str,
                        help="""Directory where to store results""")
    parser.add_argument("--snr_cut", default=None, type=float,
                        help="""Lower S/N limit.""")
    parser.add_argument('--extra_fit', default=None, type=str,
                        help="""Fit additional relation to the data to model 
                                systematic effects. Current options are 
                                'noise' and 'linear' (default = no additional fit)""")
    parser.add_argument('--fit_val', default=None, type=str,
                        help="""Which column in the catalogue to use for extra fit. Alternatively,
                                a HEALPix map can be specified, which should end in '.fits'.""")
    parser.add_argument('--healpix_mask', default=None, type=str,
                        help="Additional optional HEALPix mask.")
    parser.add_argument('--invert_mask', action='store_true',
                        help="Invert HEALPix mask specified in --healpix_mask.")
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
    parser.add_argument("--statistic", default='poisson', type=str,
                        help="Use 'poisson' or 'nb' for negative binomial")
    parser.add_argument('--keep_results', action='store_true',
                        help="""Specify if you wish to keep results from the estimator.
                                If specified the script will only run the diagnostic 
                                plots (default=False).""")

    return parser

if __name__ == '__main__':
    main()
