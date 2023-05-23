import numpy as np
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