import os
import sys
import json
from pathlib import Path

import numpy as np
from copy import deepcopy

import cycler
import healpy as hp
from healpy.newvisufunc import projview

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy.optimize import curve_fit
from scipy.stats import poisson, kstest

import astropy.units as u
from astropy.io import fits, ascii
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, Angle

import helpers

class PointingData:

    def __init__(self, catalog, catalog_name, ra_col, dec_col, flux_col, 
                 pointing_col, rms_col=None, peak_flux_col=None):
        self.catalog = catalog
        self.cat_name = catalog_name

        self.ra_col = ra_col
        self.dec_col = dec_col
        self.flux_col = flux_col
        self.peak_flux_col = peak_flux_col
        self.rms_col = rms_col
        self.pointing_col = pointing_col

        self.NSIDE = None
        self.pointings = None
        self.total_sources = len(self.catalog)

        self.out_figures = os.path.join('Figures',self.cat_name)
        if not os.path.exists(self.out_figures):
            os.mkdir(self.out_figures)

    def separate_pointings(self):
        '''
        Build table of separate pointings from the full catalog
        '''
        self.pointings = Table()

        catalog_by_pointings = self.catalog.group_by(self.pointing_col)

        self.pointings['source_id'] = catalog_by_pointings.groups.keys
        pointing_coord = [f'{pointing[1:3]} {pointing[3:5]} {pointing[5:10]} {pointing[10:13]} {pointing[13:15]} {pointing[15:17]}'
                            for pointing in self.pointings_names[pointing_col]]
        pointings_coord = SkyCoord(pointing_coord, unit=(u.hourangle, u.deg))
        self.pointings['RA'] = pointings_coord.ra.deg
        self.pointings['DEC'] = pointings_coord.dec.deg
        self.pointings['theta'], self.pointings['phi'] = helpers.RADECtoTHETAPHI(pointings_coord.ra.deg,
                                                                                 pointings_coord.dec.deg)
        l, b = helpers.equatorial_to_galactic(pointings_coord.ra.rad, pointings_coord.dec.rad)
        self.pointings['l'] = np.rad2deg(l)
        self.pointings['b'] = np.rad2deg(b)

        self.pointings['n_sources'] = np.array([len(group) for group in catalog_by_pointings.groups])
        self.total_sources = np.sum(self.pointings['n_sources'])
        if self.rms_col is not None:
            self.pointings['rms'] = np.array([group[0][self.rms_col] for group in catalog_by_pointings.groups])

    def read_pointings_catalog(self, catalog, pointing_col, ra_col, dec_col, rms_col):
        '''
        Read in prepared catalog of pointings
        '''
        self.pointings = catalog

        self.pointings['source_id'] = catalog[pointing_col]
        self.pointings['RA'] = catalog[ra_col]
        self.pointings['DEC'] = catalog[dec_col]

        self.pointings['theta'], self.pointings['phi'] = helpers.RADECtoTHETAPHI(catalog[ra_col],
                                                                                 catalog[dec_col])
        l, b = helpers.equatorial_to_galactic(np.deg2rad(catalog[ra_col]), 
                                              np.deg2rad(catalog[dec_col]))
        self.pointings['l'] = np.rad2deg(l)
        self.pointings['b'] = np.rad2deg(b)

        self.pointings['n_sources'] = np.array([len(self.catalog[self.catalog[self.pointing_col] == pointing[pointing_col]])
                                                    for pointing in catalog])
        self.total_sources = np.sum(self.pointings['n_sources'])
        if rms_col is not None:
            self.pointings['rms'] = catalog[rms_col]

    def apply_cuts_sources(self, flux_cut=None, snr_cut=None, bool_col=None, exclude_col=True):
        '''
        Apply cuts in source catalog to prepare for a dipole measurement

        Keyword arguments:
        flux_cut (float)   -- Lower flux density cut
        snr_cut (float)    -- Lower signal to noise cut
        bool_col (float)   -- Boolean column to select sources
        exclude_col (bool) -- Whether to select include true values or false
        '''
        if snr_cut:
            snr = self.catalog[self.peak_flux_col]/self.catalog[self.rms_col]
            self.catalog = self.catalog[snr > snr_cut]

        if flux_cut:
            self.catalog = self.catalog[self.catalog[self.flux_col] > flux_cut]

        if bool_col:
            if exclude_col:
                self.catalog = self.catalog[~self.catalog[bool_col]]
            else:
                self.catalog = self.catalog[self.catalog[bool_col]]

        for pointing in self.pointings:
            pointing_sources = self.catalog[self.pointing_col] == pointing['source_id']
            pointing['n_sources'] = len(self.catalog[pointing_sources])
        self.total_sources = np.sum(self.pointings['n_sources'])

        print(f'--- Catalog {self.cat_name} loaded and prepped ---')
        print(f"Number of sources after source cuts is {self.total_sources}")

    def apply_cuts_pointings(self, col, low, high, include=True):
        '''
        Apply cuts in source catalog to prepare for a dipole measurement

        Keyword arguments:
        col (str)        -- Column to select on
        low (float)      -- Lower bound of values
        high (float)     -- Upper bound of values
        include (bool)   -- Whether to select inside or outside bounds
        '''
        if include:
            self.pointings = self.pointings[np.logical_and(self.pointings[col] > low,
                                                           self.pointings[col] < high)]
        else:
            self.pointings = self.pointings[np.logical_or(self.pointings[col] < low,
                                                          self.pointings[col] > high)]

        self.total_sources = np.sum(self.pointings['n_sources'])

        print(f"Number of sources after {col} cuts is {self.total_sources} in {len(self.pointings)} pointings")

    def completeness_counts(self, completeness_col):
        '''
        Calculate effective number counts
        '''
        for pointing in self.pointings:
            pointing_sources = self.catalog[self.catalog[self.pointing_col] == pointing['source_id']]

            eff_counts = np.sum(1/pointing_sources[completeness_col])
            pointing['n_sources'] = eff_counts

    def rms_power_law(self, rms, counts, plot=False):

        def power_law(x, a, b):
            return a*x**(-b)

        popt, pcov = curve_fit(power_law, rms/self.sigma_ref, counts)
        print(f'Fit power law with index {popt[1]:.2f} and normalization {popt[0]:.2g}')
        print('Calculating mean values')

        if plot:
            rms_range = np.linspace(rms/self.sigma_ref.min(),rms/self.sigma_ref.max(),50)
            plt.plot(rms_range*self.sigma_ref, power_law(rms_range, *popt), '--k')
            plt.scatter(rms, counts, s=1, marker='+', color='k')

            plt.xlabel('$\\sigma$ (mJy/beam)')
            plt.ylabel('Counts')
            plt.xscale('log')
            plt.yscale('log')

            plt.autoscale(enable=True, axis='x', tight=True)
            plt.savefig(os.path.join(self.out_figures,f'{self.cat_name}_rms_pow.png'), dpi=300)
            plt.close()

        return popt[0], popt[1]

    def flux_power_law(self):
        '''
        Fit power law to flux distribution
        '''
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

        plt.xlabel('Flux density (Jy)')
        plt.ylabel('Counts')
        plt.legend()

        plt.savefig(os.path.join(self.out_figures, 
                                 f'{self.cat_name}_flux_dist.png'), 
                    dpi=300)
        plt.close()

    def plot_poisson(self, counts, poisson_lambda):
        bins = np.arange(counts.min(), counts.max())
        poisson_prob = poisson.pmf(bins, poisson_lambda)

        plt.hist(counts, bins=bins, density=True, 
                 color='navy', label='Data')
        plt.step(bins, poisson_prob, where='post', 
                 color='crimson', label=f'Poisson ($\lambda$={poisson_lambda:.2f})')

        plt.xlabel('Cell counts')
        plt.legend()

        plt.savefig(os.path.join(self.out_figures,
                                 f'{self.cat_name}_poisson_counts.png'), 
                    dpi=300)
        plt.close()

def catalog_data(params, flux_cut, snr_cut, extra_fit=False, completeness=None):
    '''
    Prepare catalog for a dipole measurement
    '''
    source_catalog = Table.read(params['catalog']['path'])
    data = PointingData(source_catalog, params['catalog']['name'], **params['columns'])

    # Define pointing columns if present
    if 'pointing_columns' in params:
        pointing_catalog = Table.read(params['catalog']['pointings_path'])
        data.read_pointings_catalog(pointing_catalog, **params['pointing_columns'])
    else:
        data.separate_pointings()

    # Apply source cuts if specified
    if 'source_cuts' in params:
        for cut in params['source_cuts']:
            data.apply_cuts_sources(**cut)

    # Apply pointing cuts if specified
    if 'pointing_cuts' in params:
        for cut in params['pointing_cuts']:
            data.apply_cuts_pointings(**cut)

    if extra_fit == 'noise':
        data.apply_cuts_sources(snr_cut=snr_cut)
        data.sigma_ref = np.median(data.pointings['rms'])
        print(f'Median rms value: {data.sigma_ref}')
        flux_cut = 0
    else:
        data.apply_cuts_sources(flux_cut=flux_cut)
        data.flux_power_law()
        # Apply completeness correction if specified
        if completeness:
            data.completeness_counts(completeness)

    return data, flux_cut