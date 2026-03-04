import os
import json
import numpy as np

from copy import deepcopy
from pathlib import Path
from argparse import ArgumentParser

import healpy as hp
from healpy.newvisufunc import projview
import matplotlib.pyplot as plt

from scipy.stats import poisson
from scipy.optimize import curve_fit

import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

from . import helpers

class HEALPixData:

    def __init__(self, hpx_map, nside, name, strict_mask=False):

        self.name = name
        self.nside = nside
        self.hpx_map = hpx_map

        print(f'Total Number of sources is {int(np.sum(hpx_map))}')
        self.total_sources = int(np.sum(hpx_map))

        if strict_mask:
            bad_pix = hp.get_all_neighbours(nside, np.where(self.hpx_map == 0)[0])
            self.hpx_map[bad_pix.flatten()] = 0
        self.hpx_mask = self.hpx_map == 0

        # Create HEALPix table entries
        NPIX = hp.nside2npix(self.nside)
        self.pix_table = Table()
        self.pix_table['idx'] = np.arange(NPIX)
        self.pix_table['counts'] = self.hpx_map
        self.pix_table['masked'] = self.hpx_mask

        theta, phi = hp.pix2ang(self.nside, np.arange(NPIX))
        ra, dec = helpers.THETAPHItoRADEC(theta, phi)
        self.pix_table['ra'] = ra
        self.pix_table['dec'] = dec

        ra_rad  = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)
        # Get galactic and ecliptic coordinates
        l, b       = helpers.equatorial_to_galactic(ra_rad, dec_rad)
        elon, elat = helpers.transform_spherical_coordinate_system(3*np.pi/2.,
                                                                   np.deg2rad(90-23.439),
                                                                   0., ra_rad, dec_rad)
        self.pix_table['l'] = np.rad2deg(l)
        self.pix_table['b'] = np.rad2deg(b)
        self.pix_table['elon'] = np.rad2deg(elon)
        self.pix_table['elat'] = np.rad2deg(elat)
        # Need this for CatWISE fits
        self.pix_table['abs_elat'] = np.abs(np.rad2deg(elat))

        # Other variables
        self.sigma_ref = None
        self.alpha = None
        self.x = None

        # Check if output directories exist and create them if not
        if not os.path.exists('figures'):
            os.mkdir('figures')

        self.out_figures = os.path.join('figures',self.name)
        if not os.path.exists(self.out_figures):
            os.mkdir(self.out_figures)

    def apply_mask(self, mask_file, invert_mask_file=False):
        """
        Apply cuts in the data to prepare for a dipole measurement

        Keyword arguments:
        mask_file (string)      -- File containing areas to mask
        invert_mask_file (bool) -- Invert the mask specified by the mask file
        """
        # Parse fits file
        print('Using mask file',mask_file)

        mask = hp.read_map(mask_file).astype(bool)
        if invert_mask_file:
            self.user_mask = np.invert(mask)
        else:
            self.user_mask = mask

        self.hpx_mask = np.logical_or(self.hpx_mask, self.user_mask)
        self.hpx_map[self.hpx_mask] = 0

    def apply_weights(self, weight_file):
        """
        Apply weights to HEALPix map

        Keyword arguments:
        weight_file (str) -- Name of weight file, assumed HEALPix
        """
        print('Using weight file',weight_file)

        path = Path(__file__).parents[1] / 'weights-masks' / weight_file
        weights = hp.read_map(path)
        self.hpx_map = self.hpx_map / weights

    def median_healpix_map(self, catalog, catalog_col, pix_col=None):
        """
        Make a HEALPix map of some column, taking the median for each cell

        Keyword arguments:
        col (str) -- Name of the table column to take value from
        """
        indices = hp.ang2pix(self.nside, catalog['theta'], catalog['phi'])
        idx, inverse, number_counts = np.unique(indices, return_inverse=True, 
                                                return_counts=True)

        NPIX = hp.nside2npix(self.nside)
        hpx_map = np.array([np.nanmedian(catalog[catalog_col][indices == i]) for i in range(NPIX)])

        # Add result to HEALPix table as well
        if pix_col is None:
            pix_col = catalog_col
        self.pix_table[pix_col] = hpx_map

        return hpx_map

    def show_map(self, input_map=None, name='counts', **kwargs):
        """
        Display healpix map

        Keyword arguments:
        input_map (array) -- Input HEALPix map
        name (string)     -- Name to include in plot name
        """
        if input_map is None:
            input_map = self.hpx_map
        masked = hp.ma(input_map)
        masked.mask = self.hpx_mask

        cmap = plt.cm.get_cmap("viridis").copy()
        cmap.set_bad(color='grey')

        projview(masked, cmap=cmap, **kwargs)

        outfile = f'{self.name}_{name}_nside{self.nside}.png'
        plt.savefig(os.path.join(self.out_figures, outfile), dpi=300)
        plt.close()

    def smoothed_map(self, mean, rad=1.0):
        """
        Smooth healpix map with Gaussian kernel

        Keyword arguments:
        mean (float) -- Mean value of map
        rad (float)  -- Size of kernel, in radians
        """

        def masked_smoothing(U, rad):
            """Smooth a masked map"""
            V=U.copy()
            V[U!=U]=0
            VV=hp.smoothing(V, fwhm=rad)
            W=0*U.copy()+1
            W[U!=U]=0
            WW=hp.smoothing(W, fwhm=rad)
            return VV/WW

        nan_map = self.hpx_map-mean
        nan_map[self.hpx_mask] = np.nan
        smoothed = masked_smoothing(nan_map, rad)

        smoothed_masked = hp.ma(smoothed)
        smoothed_masked.mask = self.hpx_mask

        cmap = plt.cm.get_cmap("coolwarm").copy()
        cmap.set_bad(color='grey')

        projview(smoothed_masked, cmap=cmap)
        plt.grid(True)

        outfile = f'{self.name}_smoothed{rad}rad_counts_nside{self.nside}.png'
        plt.savefig(os.path.join(self.out_figures, outfile), dpi=300)
        plt.close()

        return smoothed_masked

    def plot_poisson(self, counts, poisson_lambda):
        """
        Plot source counts distribution and Poisson distribution with same mean

        Keyword arguments:
        counts (array)         -- Array of source counts
        poisson_lambda (float) -- Lambda value for Poisson distribution
        """
        bins = np.arange(counts.min(), counts.max())
        poisson_prob = poisson.pmf(bins, poisson_lambda)

        plt.hist(counts, bins=bins, density=True, color='navy', label='Data')
        plt.step(bins, poisson_prob, where='post', color='crimson', 
                 label=f'Poisson ($\lambda$={poisson_lambda:.2f})')

        plt.xlabel('Cell counts')
        plt.legend()

        outfile = f'{self.name}_poisson_counts_nside{self.nside}.png'
        plt.savefig(os.path.join(self.out_figures,outfile), dpi=300)
        plt.close()

    def rms_power_law(self, rms, counts, plot=False):
        """
        Fit a power law between local RMS and number of sources in cells

        Keyword arguments:
        rms (array)    -- RMS values
        counts (array) -- Source count values
        plot (bool)    -- Whether to plot values and the fit
        """
        def power_law(x, a, b):
            return a*x**(-b)

        popt, pcov = curve_fit(power_law, rms/self.sigma_ref, counts)
        print(f'Fit power law with index {popt[1]:.2f} and normalization {popt[0]:.2f}')
        print('Calculating mean values')

        if plot:
            rms_range = np.linspace(rms.min()/self.sigma_ref,
                                    rms.max()/self.sigma_ref, 50)
            plt.plot(rms_range*self.sigma_ref, power_law(rms_range, *popt), '--k', 
                     label=f'$N = {{{popt[0]:.0f}}}\cdot(\\sigma/\\sigma_0)^{{-{popt[1]:.2f}}}$')
            plt.scatter(rms, counts, s=1, marker='+', color='k', label='Data')

            plt.xlabel('$\\sigma$ (mJy/beam)')
            plt.ylabel('N (counts/pixel)')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()

            plt.autoscale(enable=True, axis='x', tight=True)
            outfile = f'{self.name}_rms_pow_nside{self.nside}.png'
            plt.savefig(os.path.join(self.out_figures, outfile), dpi=300)
            plt.close()

        return popt[0], popt[1]

class CatalogData:

    def __init__(self, catalog, catalog_name, ra_col, dec_col, flux_col, 
                 rms_col=None, peak_flux_col=None):
        self.catalog = catalog
        self.name = catalog_name
        self.total_sources = len(catalog)

        self.ra_col = ra_col
        self.dec_col = dec_col
        self.flux_col = flux_col
        self.rms_col = rms_col
        self.peak_flux_col = peak_flux_col

        # HEALPix coordinates for all sources
        theta, phi = helpers.RADECtoTHETAPHI(self.catalog[self.ra_col],
                                             self.catalog[self.dec_col])
        self.catalog['theta'] = theta
        self.catalog['phi'] = phi

        # Galactic coordinates for all sources
        l, b = helpers.equatorial_to_galactic(np.deg2rad(self.catalog[self.ra_col]),
                                              np.deg2rad(self.catalog[self.dec_col]))
        self.catalog['l'] = np.rad2deg(l)
        self.catalog['b'] = np.rad2deg(b)

        # Other variables
        self.hpx_mask = None
        self.sigma_ref = None
        self.alpha = None
        self.x = None

        # Check if output directories exist and create them if not
        if not os.path.exists('figures'):
            os.mkdir('figures')

        self.out_figures = os.path.join('figures',self.name)
        if not os.path.exists(self.out_figures):
            os.mkdir(self.out_figures)

        print(f'--- Catalog {self.name} loaded ---')

    def apply_cuts(self, flux_cut=None, galactic_cut=None, snr_cut=None, 
            mask_bright=None, mask_file=None, invert_mask_file=False):
        """
        Apply cuts in the data to prepare for a dipole measurement

        Keyword arguments:
        flux_cut (float)        -- Lower flux  density cut
        galactic cut (float)    -- Galactic latitude below which sources should be cut
        mask_bright (float)     -- Mask sources and direct environment 
                                   above this flux density
        mask_file (string)      -- File containing areas to mask
        invert_mask_file (bool) -- Invert the mask specified by the mask file
        """
        if galactic_cut:
            self.catalog = self.catalog[np.logical_or(self.catalog['b'] < -1*galactic_cut,
                                                      self.catalog['b'] > galactic_cut)]

        if flux_cut:
            self.catalog = self.catalog[self.catalog[self.flux_col] > flux_cut]

        if snr_cut:
            snr = self.catalog[self.peak_flux_col]/self.catalog[self.rms_col]
            self.catalog = self.catalog[snr > snr_cut]

        if mask_bright:
            # Remove bright sources and immediate vicinity
            bright = self.catalog[self.catalog[self.flux_col] > mask_bright]

            sep_constraint = 0.3 * u.deg
            all_coord = SkyCoord(self.catalog[self.ra_col].data, 
                                 self.catalog[self.dec_col].data, 
                                 unit='deg')
            bright_coord = SkyCoord(bright[self.ra_col].data, 
                                    bright[self.dec_col].data, unit='deg')
            idxbright, idxall, d2d, d3d = all_coord.search_around_sky(bright_coord, 
                                                                      sep_constraint)

            mask = np.ones(len(self.catalog), dtype=bool)
            mask[idxall] = 0
            self.catalog = self.catalog[mask]

        if mask_file:
            if not isinstance(mask_file, list):
                mask_file = [mask_file]
                invert_mask_file = [invert_mask_file]

            for i, file in enumerate(mask_file):
                if file.endswith('.fits'):
                    # Save information for later
                    self.hpx_mask = file
                    self.hpx_mask_invert = invert_mask_file[i]

                else:
                    # Assume table
                    print('Using mask file',file)
                    path = Path(__file__).parents[1] / 'weights-masks' / file
                    mask = Table.read(path)

                    mask_sources = np.zeros(len(self.catalog))
                    for mask_area in mask:
                        ra = np.copy(self.catalog[self.ra_col])
                        dec = np.copy(self.catalog[self.dec_col])

                        if 'ra_min' in mask.colnames:
                            # Check and correct for RA wrap
                            if mask_area['ra_min'] > mask_area['ra_max']:
                                ra[ra > 180] -= 360
                                mask_area['ra_min'] -= 360

                            idx = np.logical_or.reduce((dec < mask_area['dec_min'],
                                                        dec > mask_area['dec_max'],
                                                        ra  < mask_area['ra_min'],
                                                        ra  > mask_area['ra_max']))
                        if 'radius' in mask.colnames:
                            # Check and correct for RA wrap
                            if mask_area['ra'] - mask_area['radius'] < 0:
                                ra[ra > 180] -= 360

                            source_coord = SkyCoord(ra, dec, unit='deg')
                            mask_coord = SkyCoord(mask_area['ra'], 
                                                  mask_area['dec'], unit='deg')
                            dist = mask_coord.separation(source_coord)
                            idx = dist > mask_area['radius'] * u.deg

                        mask_sources = np.logical_or(mask_sources, np.invert(idx))

                    if invert_mask_file[i]:
                        self.catalog = self.catalog[mask_sources]
                    else:
                        self.catalog = self.catalog[~mask_sources]

        print(f'Number of sources in catalog after masking is {len(self.catalog)}')

    def apply_additional_cuts(self, col, low, high, include=True):
        """
        Apply additional cuts to the catalog

        Keyword arguments:
        col (string)   -- Column name
        low (float)    -- Value of lower limit
        high (float)   -- Value of upper limit
        include (bool) -- Whether to include all values with set limits
        """
        if include:
            self.catalog = self.catalog[np.logical_and(self.catalog[col] > low,
                                                       self.catalog[col] < high)]
        else:
            self.catalog = self.catalog[np.logical_or(self.catalog[col] < low,
                                                      self.catalog[col] > high)]

        print(f"Number of sources after additional cuts is {len(self.catalog)}")

    def to_healpix_map(self, nside, strict_mask=True):
        """
        Create a healpix map of a given catalog

        Keyword arguments:
        nside (int)        -- Resolution of healpix map
        strict_mask (bool) -- Mask all pixels with a masked neighbour
        """
        indices = hp.ang2pix(nside, 
                             self.catalog['theta'], 
                             self.catalog['phi'])
        # Save index for later use
        self.catalog['hpx_idx'] = indices

        # Create map
        idx, inverse, number_counts  = np.unique(indices,
                                                 return_inverse=True,
                                                 return_counts=True)
        NPIX = hp.nside2npix(nside)
        hpx_map = np.zeros(NPIX)
        hpx_map[idx] = number_counts
        self.total_sources = int(np.sum(hpx_map))

        print(f'--- Catalog {self.name} discretized into healpix map with nside {nside} ---')
        healpix = HEALPixData(hpx_map, nside, name=self.name, strict_mask=strict_mask)

        return healpix

    def flux_power_law(self):
        """
        Fit power law to flux distribution
        """
        def power_law(x, a, b):
            return a*x**(-b)

        flux = self.catalog[self.flux_col]
        bins = np.logspace(np.log10(flux.min()), np.log10(flux.max()), 50)
        bin_means = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        counts, bins = np.histogram(flux, bins)

        popt, pcov = curve_fit(power_law, bin_means, counts)
        print(f'Fit power law with index {popt[1]:.2f} and normalization {popt[0]:.2f}')

        plt.bar(bin_means, counts, width=np.diff(bins), 
                edgecolor='k', alpha=0.8, align='center', label='Data')
        plt.plot(bins, power_law(bins, *popt), 
                 label=f'$N \propto S^{{-{popt[1]:.2f}}}$', color='k')
        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel('Flux density')
        plt.ylabel('Counts')
        plt.legend()

        plt.savefig(os.path.join(self.out_figures,
                                 f'{self.name}_flux_dist.png'), 
                    dpi=300)
        plt.close()

def catalog_data(params, nside, flux_cut, snr_cut, extra_fit=False, fit_val=None):
    """
    Prepare catalog for a dipole measurement
    """
    table = Table.read(params['catalog']['path'])
    catalog = CatalogData(table, params['catalog']['name'], **params['columns'])

    # Apply standard cuts
    if 'cuts' in params:
        catalog.apply_cuts(**params['cuts'])

    # Additional cuts
    if 'additional_cuts' in params:
        for cut in params['additional_cuts']:
            catalog.apply_additional_cuts(**cut)

    if extra_fit == 'noise':
        catalog.apply_cuts(snr_cut=snr_cut)
        data = catalog.to_healpix_map(nside)

        rms_map = data.median_healpix_map(catalog.catalog,
                                          catalog.rms_col,
                                          pix_col='rms')
        data.show_map(rms_map, name='RMS')
        data.show_map(name='counts_nocut')

        rms = rms_map[~data.hpx_mask]
        data.sigma_ref = np.nanmedian(rms[~np.isnan(rms)])
        print(f'Median RMS value {data.sigma_ref} mJy/beam')

        flux_cut = None
    else:
        catalog.apply_cuts(flux_cut=flux_cut, snr_cut=snr_cut)
        catalog.flux_power_law()

        data = catalog.to_healpix_map(nside)
        data.show_map()

        if extra_fit:
        # If additional fit value is a catalogue column, create map
            if fit_val in catalog.catalog.colnames:
                print(f'Creating median HEALPix map of {fit_val}')
                data.median_healpix_map(catalog.catalog, fit_val)

    # Check if HEALPix mask was loaded
    if catalog.hpx_mask is not None:
        data.apply_mask(catalog.hpx_mask, catalog.hpx_mask_invert)

    if 'weights' in params:
        data.apply_weights(params['weights'])
        data.show_map(name='counts_weighted')

    return data, flux_cut

def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    name = args.catalog_name
    nside = args.nside

    flux_cut = args.flux_cut
    snr_cut = args.snr_cut
    completeness = args.completeness

    path = Path(__file__).parents[1] / 'parsets' / name
    with open(path.with_suffix('.json')) as infile:
        param_sets = json.load(infile)

    for param_set in param_sets:
        if flux_cut is None:
            flux_cut_out = param_set['catalog']['default_flux_cut']
        else:
            flux_cut_out = flux_cut

        data, flux_cut_out = catalog_data(param_set, nside, flux_cut_out, snr_cut)

        label = f'{data.name}_nside{data.nside}_fluxcut{flux_cut_out:g}'
        if snr_cut is not None:
            label += f'_snrcut{snr_cut:g}'
        if completeness is not None:
            label += '_'+completeness

        # Save to HEALPix fits file
        hp.write_map(label+'.fits', data.hpx_map)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("catalog_name", type=str,
                        help="""Name of the catalog to fit dipole. Name should
                                match a corresponding json file in the 
                                parsets directory (withouth the extension).""")
    parser.add_argument("--nside", default=32, type=int,
                        help="""nside parameter for HEALPix""")
    parser.add_argument("--flux_cut", default=None, type=float,
                        help="""Lower flux density limit, will overwrite the value
                                present in the parameter json file.""")
    parser.add_argument("--snr_cut", default=None, type=float,
                        help="""Lower S/N limit.""")
    parser.add_argument('--completeness', nargs='?', const=True,
                        help="""Use completeness when looking at number counts, provided a 
                                column which stores the completeness of each source in the 
                                catalog (default = do not use completeness).""")

    return parser

if __name__ == '__main__':
    main()