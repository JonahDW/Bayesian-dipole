import os
import json
import numpy as np

from copy import deepcopy
from pathlib import Path

import healpy as hp
from healpy.newvisufunc import projview
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import poisson

import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

import simulate_observations as sim_obs
import helpers

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14

class SkyData:

    def __init__(self, catalog, catalog_name, ra_col, dec_col, flux_col, rms_col=None, peak_flux_col=None):
        self.catalog = catalog
        self.cat_name = catalog_name

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

        self.NSIDE = None
        self.hpx_map = None
        self.hpx_mask = None
        self.sigma_ref = None
        self.total_sources = len(catalog)

        # Check if output directories exist and create them if not
        if not os.path.exists('Figures'):
            os.mkdir('Figures')

        self.out_figures = os.path.join('Figures',self.cat_name)
        if not os.path.exists(self.out_figures):
            os.mkdir(self.out_figures)

        print(f'--- Catalog {self.cat_name} loaded ---')

    def apply_cuts(self, flux_cut=None, galactic_cut=None, dec_low=None,
            dec_high=None, dec_include=True, snr_cut=None, mask_bright=None, mask_file=None):
        '''
        Apply cuts in the data to prepare for a dipole measurement

        Keyword arguments:
        flux_cut (float)     -- Lower flux  density cut
        galactic cut (float) -- Galactic latitude below which sources should be cut
        dec_low (float)      -- Lower limit of declination
        dec_high (float)     -- Upper limit of declination
        dec_include (bool)   -- Whether to include or exclude the specified declination range
        mask_bright (float)  -- Mask sources and direct environment above this flux density
        mask_file (string)   -- Filename containing regions to mask
        '''
        if galactic_cut:
            self.catalog = self.catalog[np.logical_or(self.catalog['b'] < -1*galactic_cut,
                                                      self.catalog['b'] > galactic_cut)]

        if dec_low and dec_high:
            if dec_include:
                self.catalog = self.catalog[np.logical_and(self.catalog[self.dec_col] > dec_low,
                                                           self.catalog[self.dec_col] < dec_high)]
            else:
                self.pointings = self.pointings[np.logical_or(self.catalog[self.dec_col] < dec_low,
                                                              self.catalog[self.dec_col] > dec_high)]

        if flux_cut:
            self.catalog = self.catalog[self.catalog[self.flux_col] > flux_cut]

        if snr_cut:
            self.catalog = self.catalog[self.catalog[self.peak_flux_col]/self.catalog[self.rms_col] > snr_cut]

        if mask_bright:
            # Remove bright sources and immediate vicinity
            bright = self.catalog[self.catalog[self.flux_col] > mask_bright]

            sep_constraint = 0.3 * u.deg
            all_coord = SkyCoord(self.catalog[self.ra_col].data, self.catalog[self.dec_col].data, unit='deg')
            bright_coord = SkyCoord(bright[self.ra_col].data, bright[self.dec_col].data, unit='deg')
            idxbright, idxall, d2d, d3d = all_coord.search_around_sky(bright_coord, sep_constraint)

            mask = np.ones(len(self.catalog), dtype=bool)
            mask[idxall] = 0
            self.catalog = self.catalog[mask]

        if mask_file:
            path = Path(__file__).parent / 'parsets' / mask_file
            with open(path) as infile:
                mask = json.load(infile)

            for mask_area in mask:
                ra = np.copy(self.catalog[self.ra_col])
                dec = np.copy(self.catalog[self.dec_col])

                # Check and correct for RA wrap
                if mask[mask_area]['ra_min'] > mask[mask_area]['ra_max']:
                    ra[ra > 180] -= 360
                    mask[mask_area]['ra_min'] -= 360

                idx = np.logical_or.reduce((dec < mask[mask_area]['dec_min'],
                                            dec > mask[mask_area]['dec_max'],
                                            ra < mask[mask_area]['ra_min'],
                                            ra > mask[mask_area]['ra_max']))
                self.catalog = self.catalog[idx]

        print(f'Number of sources in catalog after masking is {len(self.catalog)}')

    def to_healpix_map(self, NSIDE, strict_mask=True):
        '''
        Create a healpix map of a given catalog

        Keyword arguments:
        NSIDE (int)        -- Resolution of healpix map
        strict_mask (bool) -- Mask all pixels with a masked neighbour
        '''
        self.NSIDE = NSIDE

        indices = hp.ang2pix(self.NSIDE, self.catalog['theta'], self.catalog['phi'])
        idx, inverse, number_counts  = np.unique(indices, return_inverse=True, return_counts=True)

        NPIX = hp.nside2npix(self.NSIDE)
        self.hpx_map = np.zeros(NPIX)
        self.hpx_map[idx] = number_counts

        if strict_mask:
            bad_pix = hp.get_all_neighbours(NSIDE, np.where(self.hpx_map == 0)[0])
            self.hpx_map[bad_pix.flatten()] = 0
        self.hpx_mask = self.hpx_map == 0

        print(f'--- Catalog {self.cat_name} discretized into healpix map with NSIDE {NSIDE} ---')
        print(f'Number of sources after mapping is {int(np.sum(self.hpx_map))}')
        self.total_sources = int(np.sum(self.hpx_map))

    def median_healpix_map(self, NSIDE):
        '''
        Make a HEALPix map of the RMS, taking the median for each cell

        Keyword arguments:
        NSIDE (int)       -- Resolution of HEALPix map
        '''
        indices = hp.ang2pix(NSIDE, self.catalog['theta'], self.catalog['phi'])
        idx, inverse, number_counts  = np.unique(indices, return_inverse=True, return_counts=True)

        NPIX = hp.nside2npix(NSIDE)
        hpx_map = np.array([np.nanmedian(self.catalog[self.rms_col][indices == i]) for i in range(NPIX)])

        self.sigma_ref = np.nanmedian(hpx_map[~np.isnan(hpx_map)])
        print(f'Median RMS value {self.sigma_ref} mJy/beam')

        return hpx_map

    def show_map(self, input_map=None, name='counts', **kwargs):
        '''
        Display healpix map

        Keyword arguments:
        input_map (array) -- Input HEALPix map
        name (string)     -- Name of quantity to include in plot name
        '''
        if input_map is None:
            input_map = self.hpx_map
        masked = hp.ma(input_map)
        masked.mask = self.hpx_mask

        cmap = plt.cm.get_cmap("viridis").copy()
        cmap.set_bad(color='grey')

        projview(masked, cmap=cmap, **kwargs)
        plt.savefig(os.path.join(self.out_figures,
                        f'{self.cat_name}_{name}_NSIDE{self.NSIDE}.png'),
                    dpi=300)
        plt.close()

    def smoothed_map(self, mean):

        def masked_smoothing(U, rad=5.0):
            V=U.copy()
            V[U!=U]=0
            VV=hp.smoothing(V, fwhm=rad)
            W=0*U.copy()+1
            W[U!=U]=0
            WW=hp.smoothing(W, fwhm=rad)
            return VV/WW

        nan_map = self.hpx_map
        nan_map[self.hpx_mask] = np.nan
        smoothed = masked_smoothing(nan_map, rad=1.0)

        smoothed_masked = hp.ma(smoothed)
        smoothed_masked.mask = self.hpx_mask

        cmap = plt.cm.get_cmap("coolwarm").copy()
        cmap.set_bad(color='grey')

        projview(smoothed_masked, cmap=cmap)
        plt.savefig(os.path.join(self.out_figures,
                        f'{self.cat_name}_smoothed_counts_NSIDE{self.NSIDE}.png'),
                    dpi=300)
        plt.close()

    def rms_power_law(self, fit_rms, fit_sources, plot=False):
        '''
        Fit a power law between local RMS and number of sources in cells

        Keyword arguments:
        fit_rms (array)     -- RMS values
        fit_sources (array) -- Source count values
        plot (bool)         -- Whether to plot values and the fit
        '''
        def power_law(x, a, b):
            return a*x**(-b)

        popt, pcov = curve_fit(power_law, fit_rms/self.sigma_ref, fit_sources)
        print(f'Fit power law with index {popt[1]:.2f} and normalization {popt[0]:.2f}')
        print('Calculating mean values')

        if plot:
            rms_range = np.linspace(fit_rms.min()/self.sigma_ref,fit_rms.max()/self.sigma_ref,50)
            plt.plot(rms_range*self.sigma_ref, power_law(rms_range, *popt), '--k', 
                     label=f'$N = {{{popt[0]:.0f}}}\cdot(\\sigma/\\sigma_0)^{{-{popt[1]:.2f}}}$')
            plt.scatter(fit_rms, fit_sources, s=1, marker='+', color='k', label='Data')

            plt.xlabel('$\\sigma$ (mJy/beam)')
            plt.ylabel('N (counts/pixel)')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()

            plt.autoscale(enable=True, axis='x', tight=True)
            plt.savefig(os.path.join(self.out_figures,
                            f'{self.cat_name}_rms_pow_NSIDE{self.NSIDE}.png'),
                        dpi=300)
            plt.close()

        return popt[0], popt[1]

    def plot_poisson(self, counts, poisson_lambda):
        '''
        Plot source counts distribution and Poisson distribution with same mean

        Keyword arguments:
        counts (array)         -- Array of source counts
        poisson_lambda (float) -- Lambda value for Poisson distribution
        '''
        bins = np.arange(counts.min(), counts.max())
        poisson_prob = poisson.pmf(bins, poisson_lambda)

        plt.hist(counts, bins=bins, density=True, color='navy', label='Data')
        plt.step(bins, poisson_prob, where='post', color='crimson', 
                 label=f'Poisson ($\lambda$={poisson_lambda:.2f})')

        plt.xlabel('Cell counts')
        plt.legend()

        plt.savefig(os.path.join(self.out_figures,
                        f'{self.cat_name}_poisson_counts_NSIDE{self.NSIDE}.png'), 
                    dpi=300)
        plt.close()

def simulated_data(NSIDE, results_dir, flux_cut, snr_cut, fit_noise=False):
    '''
    Simulate a dataset and run likelihood
    '''
    sim_param_file = 'parsets/simulation.json'
    with open(sim_param_file) as f:
        sim_params = json.load(f)

    sim_cat_file = os.path.join(results_dir,'catalog_simulation')
    name = 'Simulation'

    if os.path.exists(sim_cat_file+'.fits'):
        sim_catalog = Table.read(sim_cat_file+'.fits')
    else:
        dipole_params = sim_params['dipole']
        survey_params = sim_params['contiguous']

        dipole = (dipole_params['beta'], dipole_params['ra'], dipole_params['dec'])

        ra, dec = sim_obs.sample_uniform_sky(survey_params['N'])
        ra_false, dec_false = sim_obs.sample_uniform_sky(int(survey_params['false_detection_frac']
                                                             * survey_params['N']))

        flux = sim_obs.sample_power_law_flux(survey_params['N'],
                                         pow_norm=survey_params['pow_norm'],
                                         pow_index=survey_params['pow_index'])
        flux_false = sim_obs.sample_power_law_flux(int(survey_params['false_detection_frac']
                                                       * survey_params['N']),
                                               pow_norm=survey_params['pow_norm'],
                                               pow_index=survey_params['pow_index'])

        ra, dec, flux = sim_obs.apply_dipole(dipole, ra, dec, flux, survey_params['alpha'])

        if survey_params['rms_variation'] == 'RACS':
            path = Path(__file__).parent / 'parsets' / 'RACS.json'
            with open(path) as infile:
                racs_params = json.load(infile)[0]

            # Get RMS noise from RACS
            RACS_catalog = Table.read(racs_params['catalog']['path'])
            RACS = SkyData(RACS_catalog, racs_params['catalog']['name'], **racs_params['columns'])
            RACS.apply_cuts(**racs_params['cuts'])
            median_rms = RACS.median_healpix_map(NSIDE)

            theta, phi = helpers.RADECtoTHETAPHI(np.rad2deg(ra), np.rad2deg(dec))
            pix = hp.ang2pix(NSIDE, theta, phi)
            rms = median_rms[pix]

            theta_false, phi_false = helpers.RADECtoTHETAPHI(np.rad2deg(ra_false), np.rad2deg(dec_false))
            pix_false = hp.ang2pix(NSIDE, theta_false, phi_false)
            rms_false = median_rms[pix_false]
        else:
            rms = survey_params['rms'] * (1 - survey_params['rms_variation']*np.cos(ra))
        rms_noise = np.random.normal(0, rms)

        sim_catalog = Table()
        sim_catalog['ra'] = np.rad2deg(ra)
        sim_catalog['dec'] = np.rad2deg(dec)
        sim_catalog['intrinsic_flux'] = flux # mJy
        sim_catalog['observed_flux'] = flux + rms_noise # mJy
        sim_catalog['rms'] = rms

        # False sources
        if survey_params['false_detection_frac'] > 0:
            false_catalog = Table()
            false_catalog['ra'] = np.rad2deg(ra_false)
            false_catalog['dec'] = np.rad2deg(dec_false)
            false_catalog['intrinsic_flux'] = flux_false # mJy
            false_catalog['observed_flux'] = flux_false # mJy
            false_catalog['rms'] = rms_false

            sim_catalog = vstack([sim_catalog, false_catalog])

        # Retain only sources with 5 sigma
        sim_catalog = sim_catalog[sim_catalog['rms'] != np.nan]
        sim_catalog = sim_catalog[sim_catalog['observed_flux']/sim_catalog['rms'] > 5]
        sim_catalog.write(sim_cat_file+'.fits')

    dipole_amplitude = ( (2 + sim_params['contiguous']['pow_index'] 
                        * (1 + sim_params['contiguous']['alpha'])) 
                        * sim_params['dipole']['beta'] )

    sim_data = SkyData(sim_catalog, catalog_name=name,
                  ra_col='ra', dec_col='dec',
                  flux_col='observed_flux', 
                  peak_flux_col='observed_flux',
                  rms_col='rms')

    if fit_noise:
        sim_data.apply_cuts(snr_cut=snr_cut)
        sim_data.to_healpix_map(NSIDE)

        median_rms = sim_data.median_healpix_map(NSIDE)
        sim_data.show_map(median_rms, name='RMS')
        sim_data.show_map(name='counts_nocut')

        fit_sources = sim_data.hpx_map[~sim_data.hpx_mask]
        median_rms = median_rms[~sim_data.hpx_mask]
        sim_data.rms_power_law(median_rms, fit_sources, plot=True)

        flux_cut = 0
    else:
        sim_data.apply_cuts(flux_cut=flux_cut)
        sim_data.to_healpix_map(NSIDE)
        sim_data.show_map()

    return sim_data, flux_cut, dipole_amplitude

def catalog_data(params, NSIDE, flux_cut, snr_cut, fit_noise=False):
    '''
    Prepare catalog for a dipole measurement
    '''
    catalog = Table.read(params['catalog']['path'])
    data = SkyData(catalog, params['catalog']['name'], **params['columns'])
    data.apply_cuts(**params['cuts'])

    if fit_noise:
        data.apply_cuts(snr_cut=snr_cut)
        data.to_healpix_map(NSIDE)

        rms = data.median_healpix_map(NSIDE)
        data.show_map(rms, name='RMS')
        data.show_map(name='counts_nocut')

        flux_cut = 0
    else:
        data.apply_cuts(flux_cut=flux_cut)

        data.to_healpix_map(NSIDE)
        data.show_map()

    return data, flux_cut