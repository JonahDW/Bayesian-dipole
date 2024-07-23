import os
import sys
import json
from pathlib import Path

import numpy as np
import healpy as hp

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, Angle

import helpers

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

def sample_power_law_flux(N, pow_norm, pow_index):
    '''
    Simulate observation with a power law flux distribution

    Keyword arguments:
    N (int)           -- Number of sources to simulate
    pow_norm (float)  -- Normalization of the power law
    pow_index (float) -- Power law index

    Return:
    flux (array) -- Simulated observations
    '''
    uniform = np.random.uniform(low=0, high=1, size=N)
    flux = pow_norm*(1-uniform)**(-1/pow_index)

    return flux

def apply_dipole(dipole, ra, dec, flux, alpha):
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

def simulate_pointings():
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

    # Define columns for source catalog
    s_ra_col       = sim_params['columns']['ra_col']
    s_dec_col      = sim_params['columns']['dec_col']
    s_rms_col      = sim_params['columns']['rms_col']
    s_flux_col     = sim_params['columns']['flux_col']
    s_pointing_col = sim_params['columns']['pointing_col']

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
        pointing_cat[s_pointing_col] = [pointing[p_pointing_col]]*len(ra)
        pointing_cat[s_ra_col]       = ra
        pointing_cat[s_dec_col]      = dec
        pointing_cat[s_rms_col]      = rms
        pointing_cat[s_flux_col]     = flux + rms_noise
        pointing_cat['intrinsic_flux'] = flux

        # Do selections
        pointing_cat = pointing_cat[pointing_cat[s_rms_col] != np.nan]
        pointing_cat = pointing_cat[pointing_cat[s_flux_col]/pointing_cat[s_rms_col] > 5]

        sim_catalog = vstack([sim_catalog, pointing_cat])

    sim_cat_file = Path(__file__).parent / 'data' / 'sim_pointings'
    print(f'Simulated catalogue written to {sim_cat_file}.fits')
    sim_catalog.write(str(sim_cat_file)+'.fits', overwrite=True)

    dipole_amplitude = ( (2 + survey_params['pow_index'] 
                        * (1 + survey_params['alpha'])) 
                        * sim_params['dipole']['beta'] )
    print(f'Expected measured dipole amplitude: {dipole_amplitude}')

def simulate_sky():
    '''
    Simulate a dataset and run likelihood
    '''
    print('Simulating a source catalogue')
    sim_param_file = Path(__file__).parent / 'parsets' / 'sim-sky.json'
    with open(sim_param_file) as f:
        sim_params = json.load(f)

    dipole_params = sim_params['dipole']
    dipole = (dipole_params['beta'], dipole_params['ra'], dipole_params['dec'])

    survey_params = sim_params['contiguous']

    # Simulate sources
    ra, dec = sample_uniform_sky(survey_params['N'])
    flux = sample_power_law_flux(survey_params['N'],
                                 pow_norm=survey_params['pow_norm'],
                                 pow_index=survey_params['pow_index'])

    # Simulate false detections
    if survey_params['false_detection_frac'] > 0:
        ra_false, dec_false = sample_uniform_sky(int(survey_params['false_detection_frac']
                                                             * survey_params['N']))
        flux_false = sample_power_law_flux(int(survey_params['false_detection_frac']
                                                       * survey_params['N']),
                                               pow_norm=survey_params['pow_norm'],
                                               pow_index=survey_params['pow_index'])

    ra, dec, flux = sim_obs.apply_dipole(dipole, ra, dec, flux, survey_params['alpha'])

    # Define noise structure of the catalogue
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

    # Create catalogue
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

    sim_param_file = Path(__file__).parent / 'data' / 'sim_sky'
    print(f'Simulated catalogue written to {sim_param_file}.fits')
    sim_catalog.write(sim_cat_file+'.fits')

    dipole_amplitude = ( (2 + sim_params['contiguous']['pow_index'] 
                        * (1 + sim_params['contiguous']['alpha'])) 
                        * sim_params['dipole']['beta'] )
    print(f'Expected measured dipole amplitude: {dipole_amplitude}')

def main():

    mode = sys.argv[1]
    if 'pointing' in mode:
        simulate_pointings()
    if 'sky' in mode:
        simulate_sky()

if __name__ == '__main__':
    main()