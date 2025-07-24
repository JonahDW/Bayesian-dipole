import os
import sys
import json

from pathlib import Path
from configparser import ConfigParser
from argparse import ArgumentParser

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, Angle

from utils import helpers

def sample_uniform_sky(N):
    '''
    Simulate observation with a uniform distribution on the sky

    Keyword arguments:
    N (int) -- Number of sources to simulate

    Return:
    ra, dec (arrays) -- Simulated observations
    '''
    ra = 2 * np.pi * np.random.uniform(size=N)
    dec = np.arcsin(2* (np.random.uniform(size=N) - 0.5))

    return ra, dec

def sample_pointing(N, radius):
    '''
    Samples source in a circular image
    '''
    r = np.sqrt(np.random.uniform(size=N))
    theta = 2*np.pi*np.random.uniform(size=N)

    x = radius * r * np.cos(theta)
    y = radius * r * np.sin(theta)

    return x, y

def sample_power_law_flux(N, min_flux, pow_index, debug):
    '''
    Simulate observation with a power law flux distribution

    Keyword arguments:
    N (int)           -- Number of sources to simulate
    pow_norm (float)  -- Normalization of the power law
    pow_index (float) -- Power law index

    Return:
    flux (array) -- Simulated observations
    '''
    uniform = np.random.uniform(size=N)
    # Generate samples, this works, don't ask me how
    flux = min_flux * (1 - uniform)**(-1/pow_index)

    if debug:
        plt.hist(flux, bins=np.logspace(np.log10(min_flux),1,50))
        plt.xscale('log')
        plt.show()

    return flux

def apply_residual_dipole(dipole, ra, dec):
    '''
    Apply a residual dipole to a uniform dataset
    by 'moving' sources to the other hemisphere with
    a probability according to dipole amplitude
    '''
    amp, dipole_ra, dipole_dec = dipole

    lon, lat = helpers.equatorial_to_dipole(ra, dec, dipole_ra, dipole_dec)
    swap = ( 1 - amp*np.cos(0.5*np.pi - lat) ) * np.random.uniform(size=len(lon))
    lat[swap > 1] = -lat[swap > 1]

    ra, dec = helpers.dipole_to_equatorial(lon, lat, dipole_ra, dipole_dec)

    return ra, dec

def apply_kinematic_dipole(dipole, ra, dec, flux, alpha):
    '''
    Apply a dipole with some amplitude to a uniform dataset
    '''
    velocity, dipole_ra, dipole_dec = dipole

    lon, lat = helpers.equatorial_to_dipole(ra, dec, dipole_ra, dipole_dec)
    angle_dipole = 0.5*np.pi - lat

    flux_dipole = helpers.flux_dipole(velocity, alpha, angle_dipole)
    flux *= flux_dipole
    new_angle = helpers.angle_dipole(angle_dipole, velocity)

    lat = 0.5*np.pi - new_angle
    ra, dec = helpers.dipole_to_equatorial(lon, lat, dipole_ra, dipole_dec)

    return ra, dec, flux

def simulate_pointings(debug):
    '''
    Simulate a dataset with observed pointings
    '''
    print('Simulating a catalogue of pointings')
    sim_param_file = Path(__file__).parent / 'parsets' / 'sim-pointings.json'
    with open(sim_param_file) as f:
        sim_params = json.load(f)[0]

    # Open pointings catalogue and define column names
    pointings = Table.read(sim_params['catalog']['pointings_path'])
    p_ra_col       = sim_params['pointing_columns']['ra_col']
    p_dec_col      = sim_params['pointing_columns']['dec_col']
    p_rms_col      = sim_params['pointing_columns']['rms_col']
    p_pointing_col = sim_params['pointing_columns']['pointing_col']

    # Define input dipole
    dipole_params = sim_params['dipole']
    dipole = (dipole_params['beta'], dipole_params['ra'], dipole_params['dec'])

    # Define noise structure of catalogue
    survey_params = sim_params['simulate']
    constant_rms = True
    if isinstance(survey_params['rms'], float):
        rms_pointing = [survey_params['rms']]*len(pointings)
    elif survey_params['rms'] == 'pointing':
        rms_pointing = pointings[p_rms_col]
    else:
        constant_rms = False
        # Read rms map and header information
        path = Path(__file__).parent / 'parsets' / survey_params['rms']

        noise = fits.open(path)[0]
        noise_map = np.squeeze(noise.data)
        im_cen = noise.header['NAXIS1']/2
        pix_size = max(noise.header['CDELT1'], noise.header['CDELT2'])

    # Generate number of sources for each pointing
    npointings = len(pointings)
    nperpointing = int(survey_params['N']/npointings)

    # Simulate sources and create catalogue
    sim_catalog = Table()
    for i, pointing in enumerate(pointings):

        flux = sample_power_law_flux(nperpointing, pow_norm=survey_params['pow_norm'],
                                     pow_index=survey_params['pow_index'])
        # Generate source positions
        x, y = sample_pointing(nperpointing, survey_params['outer_radius'])
        ra = pointing[p_ra_col] + ( x / np.cos(np.deg2rad(pointing[p_dec_col])) )
        dec = pointing[p_dec_col] + y

        # Ensure proper boundaries
        ra[ra < 0.] += 360.
        ra[ra > 360.] -= 360
        ra = np.deg2rad(ra)
        dec = np.deg2rad(dec)

        # Apply dipole
        ra, dec, flux = apply_dipole(dipole, ra, dec, flux, survey_params['alpha'])
        ra = np.rad2deg(ra)
        dec = np.rad2deg(dec)

        # Remove outer radius of pointing
        pointing_centre = SkyCoord(pointing[p_ra_col], pointing[p_dec_col], unit='deg')
        source_coords = SkyCoord(ra, dec, unit='deg')
        separation = pointing_centre.separation(source_coords).deg

        ra   = ra[separation < survey_params['inner_radius']]
        dec  = dec[separation < survey_params['inner_radius']]
        flux = flux[separation < survey_params['inner_radius']]

        # Assign rms based on source positions
        if constant_rms:
            rms = [rms_pointing[i]]*len(ra)
        else:
            x = ra - pointing[p_ra_col]
            y = dec - pointing[p_dec_col]

            # Redo boundaries once again
            x[x > 180] -= 360
            x[x < -180] += 360
            x = x * np.cos(np.deg2rad(pointing[p_dec_col]))

            im_x = im_cen + x / pix_size
            im_y = im_cen + y / pix_size

            pointing_noise = noise_map*pointing[p_rms_col]
            rms = pointing_noise[im_y.astype(int),im_x.astype(int)]
        rms_noise = np.random.normal(0, rms)

        # Create catalogue of sources in pointing
        pointing_cat = Table()
        pointing_cat['pointing'] = [pointing[p_pointing_col]]*len(ra)
        pointing_cat['ra']       = ra
        pointing_cat['dec']      = dec
        pointing_cat['rms']      = rms
        pointing_cat['flux']     = flux + rms_noise
        pointing_cat['intrinsic_flux'] = flux

        # Do selections
        pointing_cat = pointing_cat[pointing_cat['rms'] != np.nan]
        pointing_cat = pointing_cat[pointing_cat['flux']/pointing_cat['rms'] > 5]

        sim_catalog = vstack([sim_catalog, pointing_cat])

    sim_cat_file = Path(__file__).parent / 'data' / 'sim_pointings'
    print(f'Simulated catalogue written to {sim_cat_file}.fits')
    sim_catalog.write(str(sim_cat_file)+'.fits', overwrite=True)

    dipole_amplitude = ( (2 + survey_params['pow_index'] 
                        * (1 + survey_params['alpha'])) 
                        * sim_params['dipole']['beta'] )
    print(f'Expected measured dipole amplitude: {dipole_amplitude}')

def simulate_sky(debug):
    '''
    Simulate a dataset and run likelihood
    '''
    print('Simulating a source catalogue')
    sim_conf_file = Path(__file__).parent / 'parsets' / 'sim-sky.cfg'

    config = ConfigParser()
    config.read(sim_conf_file)

    # Simulate sources
    sim_params = config['simulate']
    nsources   = int(sim_params.getfloat('N'))
    min_flux   = sim_params.getfloat('min_flux')
    pow_index  = sim_params.getfloat('flux_pow_index')
    alpha      = sim_params.getfloat('spectral_index')

    ra, dec = sample_uniform_sky(nsources)
    flux    = sample_power_law_flux(nsources, min_flux, pow_index, debug)

    # Simulate false detections
    if config.has_option('simulate','false_detection_frac'):
        false_frac = survey_params.getfloat('false_detection_frac')
        nsources_false = int(false_frac * nsources)
        ra_false, dec_false = sample_uniform_sky(nsources_false)
        flux_false = sample_power_law_flux(nsources_false, min_flux, pow_index, debug)

    # Simulate dipole
    dipole_params = config['dipole']
    if dipole_params.getboolean('residual'):
        resid_dipole = (dipole_params.getfloat('residual_amp'), 
                        dipole_params.getfloat('residual_ra'), 
                        dipole_params.getfloat('residual_dec'))
        ra, dec = apply_residual_dipole(resid_dipole, ra, dec)
    if dipole_params.getboolean('kinematic'):
        kin_dipole = (dipole_params.getfloat('beta'), 
                      dipole_params.getfloat('kinematic_ra'), 
                      dipole_params.getfloat('kinematic_dec'))
        ra, dec, flux = apply_kinematic_dipole(kin_dipole, ra, dec, flux, alpha)

    # Define noise structure of the catalogue
    try:
        rms = sim_params.getfloat('rms')
        rms_sources = [rms]*len(ra)
    except ValueError:
        print('RMS can not be interpreted as float, assuming input file')
        # Assume input file is HEALPix map
        path = Path(__file__).parent / survey_params['rms']
        hpx_rms_map, header = hp.read_map(path, h=True)
        header = dict(header)
        NSIDE = header['NSIDE']

        theta, phi = helpers.RADECtoTHETAPHI(np.rad2deg(ra), np.rad2deg(dec))
        pix = hp.ang2pix(NSIDE, theta, phi)
        rms_sources = hpx_rms_map[pix]

        theta_false, phi_false = helpers.RADECtoTHETAPHI(np.rad2deg(ra_false), np.rad2deg(dec_false))
        pix_false = hp.ang2pix(NSIDE, theta_false, phi_false)
        rms_false = hpx_rms_map[pix_false]

    rms_noise = np.random.normal(0, rms_sources)

    # Create catalogue
    sim_catalog = Table()
    sim_catalog['ra']   = np.rad2deg(ra)
    sim_catalog['dec']  = np.rad2deg(dec)
    sim_catalog['flux'] = flux + rms_noise # mJy
    sim_catalog['rms']  = rms
    sim_catalog['intrinsic_flux'] = flux # mJy

    # False sources
    if config.has_option('simulate','false_detection_frac'):
        false_catalog = Table()
        false_catalog['ra']   = np.rad2deg(ra_false)
        false_catalog['dec']  = np.rad2deg(dec_false)
        false_catalog['flux'] = flux_false # mJy
        false_catalog['rms']  = rms_false
        false_catalog['intrinsic_flux'] = flux_false # mJy

        sim_catalog = vstack([sim_catalog, false_catalog])

    # Retain only sources with 5 sigma
    sim_catalog = sim_catalog[sim_catalog['rms'] != np.nan]
    sim_catalog = sim_catalog[sim_catalog['flux']/sim_catalog['rms'] > 5]

    sim_cat_file = config['global']['outfile']
    print(f'--> Simulated catalogue written to {sim_cat_file}')
    sim_catalog.write(sim_cat_file, overwrite=True)

    dipole_amplitude = ( (2 + pow_index * (1 + alpha)) 
                        * dipole_params.getfloat('beta') )
    print(f'Expected kinematic dipole amplitude: {dipole_amplitude}')
    print(f"Expected total dipole amplitude: {dipole_params.getfloat('residual_amp')+dipole_amplitude}")

def main():

    parser = ArgumentParser()

    parser.add_argument("catalog_type", type=str,
                        help="""Type of catalogue to produce, can be either 'sky', generating
                                a contiguous sky survey, or 'pointing', generating a pointing
                                based catalogue.""")
    parser.add_argument("--debug", action='store_true',
                        help="Display debug messages or plots.")

    args = parser.parse_args()
    mode = args.catalog_type
    debug = args.debug

    if 'pointing' in mode:
        simulate_pointings(debug)
    if 'sky' in mode:
        simulate_sky(debug)

if __name__ == '__main__':
    main()