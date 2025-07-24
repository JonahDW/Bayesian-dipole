import os
import json
import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import binned_statistic

from pathlib import Path
from argparse import ArgumentParser

from utils.process_catalog import SkyData

def chi_square(x, x_model, k):
    """Calulate chi-square statistic"""
    chi2 = np.sum( ( x - x_model )**2 / x_model)
    dof = len(x) - k
    return chi2, dof

def pearson_correlation(x, y):
    """Calulate Pearson's correlation statistic"""
    upper = np.sum((x-np.mean(x))*(y-np.mean(y)))
    lower = np.sqrt(np.sum((x-np.mean(x))**2)) * np.sqrt(np.sum((y-np.mean(y))**2))
    return upper/lower

def get_haslam(nside, haslam_path):
    """Load Haslam map and convert to appropriate resolution"""
    haslam_highres = hp.read_map(haslam_path)
    haslam_gal = hp.ud_grade(haslam_highres, nside)

    r = hp.rotator.Rotator(coord=['G','C'])
    haslam = r.rotate_map_pixel(haslam_gal)

    return haslam

def plot_correlation(data, col, tag):
    """
    Plot and calculate key statistics for correlation between
    a given data column and number counts

    Keyword arguments:
    data (class) -- SkyData instance
    col (str)    -- Column name in HEALPix table
    tag (str)    -- Additional tag for plot name
    """
    col_hpx = data.hpx_table[col][~data.hpx_mask]
    counts = data.hpx_map[~data.hpx_mask]

    mean_count = np.mean(counts)
    chi2, dof = chi_square(counts, mean_count, k=1)
    red_chi2 = chi2/dof

    r = pearson_correlation(col_hpx, counts)

    mean_counts, edges, _ = binned_statistic(col_hpx, counts, 'mean', bins=15)
    std_counts, edges, _ = binned_statistic(col_hpx, counts, 'std', bins=15)
    bin_means = (edges[1:] + edges[:-1])/2

    plt.scatter(col_hpx, counts, color='k', alpha=0.5, s=2, label='HEALPix pixels')
    plt.errorbar(bin_means, mean_counts, yerr=std_counts, capsize=2, ls='none',
                 color='crimson', marker='o', markersize=5, label='Binned mean')
    plt.axhline(mean_count, 0, 1, color='grey', label='Mean')

    plt.text(0.05, 0.95, s=f'$\\chi^2_{{\\nu}} = {red_chi2:.2f}$', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, s=f'$\\rho = {r:.2f}$', transform=plt.gca().transAxes)

    plt.xlabel(col)
    plt.ylabel('Counts')

    outfile = f'{data.cat_name}_{col}_counts_NSIDE{data.NSIDE}_{tag}.png'
    plt.savefig(os.path.join(data.out_figures, 'systematics-check', outfile), dpi=300)
    plt.close()

def systematics_check(data, extra_columns, tag, haslam_map=None):
    """
    Check for systematics or correlation between a number of key
    observables and number counts

    Keyword arguments:
    data (class)         -- SkyData instance
    extra_columns (list) -- Additional catalogue columns to check
    tag (str)            -- Additional tag for plot names
    haslam_map (str)     -- Path to Haslam map, if using
    """

    standard_columns = ['ra', 'dec', 'l', 'b', 'elon', 'elat']

    if haslam_map is not None:
        # Get Haslam
        haslam = get_haslam(data.NSIDE, haslam_map)
        data.hpx_table['haslam_temp'] = haslam
        standard_columns += ['haslam_temp']

    # Create HEALPix maps of additional columns
    if extra_columns is not None:
        for ecol in extra_columns:
            data.median_healpix_map(ecol)
        columns = standard_columns + extra_columns
    else:
        columns = standard_columns

    for col in columns:
        plot_correlation(data, col, tag)

def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    name = args.catalog_name
    nside = args.nside
    flux_cut = args.flux_cut
    columns = args.columns
    haslam_map = args.haslam_map

    # Open parameter file
    path = Path(__file__).parent / 'parsets' / name
    with open(path.with_suffix('.json')) as infile:
        params = json.load(infile)[0]

    catalog = Table.read(params['catalog']['path'])
    data = SkyData(catalog, params['catalog']['name'], **params['columns'])
    data.to_healpix_map(NSIDE=nside)

    outdir = os.path.join(data.out_figures, 'systematics-check')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Initial systematics check without mask
    systematics_check(data, columns, tag='nocuts', haslam_map=haslam_map)

    # Apply standard cuts
    data.apply_cuts(flux_cut=flux_cut)
    if 'cuts' in params:
        data.apply_cuts(**params['cuts'])
    # Additional cuts
    if 'additional_cuts' in params:
        for cut in params['additional_cuts']:
            data.apply_additional_cuts(**cut)
    data.to_healpix_map(NSIDE=nside)

    # Now with masks
    systematics_check(data, columns, tag='cuts', haslam_map=haslam_map)

    # Create smoothed maps at two different resolutions
    mean = np.mean(data.hpx_map[~data.hpx_mask])
    smoothed = data.smoothed_map(mean, rad=0.05)
    smoothed = data.smoothed_map(mean, rad=1)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("catalog_name", type=str,
                        help="""Name of the catalog to fit dipole. Name should
                                match a corresponding json file in the 
                                parsets directory (withouth the extension).""")
    parser.add_argument("--nside", default=32, type=int,
                        help="NSIDE parameter for HEALPix map")
    parser.add_argument("--flux_cut", default=None, type=float,
                        help="""Lower flux density limit, will overwrite the value
                                present in the parameter json file.""")
    parser.add_argument("--columns", default=None, nargs='+',
                        help="Extra columns to fit for correlations.")
    parser.add_argument("--haslam_map", default=None, 
                        type=str, help="Path to Haslam map to correlate with data.")

    return parser

if __name__ == '__main__':
    main()