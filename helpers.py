import os
import matplotlib.pyplot as plt

import healpy as hp
import numpy as np
import corner

def RADECtoTHETAPHI(RA, DEC):
    '''Convert RA and DEC (in degrees) to healpix sky coordinates'''
    theta = 0.5*np.pi - np.deg2rad(DEC)
    phi = np.deg2rad(RA)
    return theta, phi

def THETAPHItoRADEC(theta, phi):
    '''Convert healpix sky coordinates to RA and DEC (in degrees)'''
    RA = np.rad2deg(phi)
    DEC = np.rad2deg(0.5*np.pi - theta)
    return RA, DEC

def great_circle_distance(x0, y0, x1, y1):
    '''
    Calculate great circle distance between two (sets of) points in radians

    Keyword arguements:
    x0, y0 : Longitude and latitude of first (set of) point(s)
    x1, y1 : Longitude and latitude of second (set of) point(s)
    '''
    return np.arccos(np.sin(y0)*np.sin(y1)+np.cos(y0)*np.cos(y1)*np.cos(np.abs(x0-x1)))

def transform_spherical_coordinate_system(x0, y0, a0, x, y, inverse=False):
    '''
    General function for transforming between two systems of spherical coordinates
    for an inverse transformation, swap the longitude values y0 and b0

    Keyword arguments:
    x0, y0 : Longitude and latitude of the pole in coordinate system one
    a0 : Longitude of the pole in coordinate system two
    x, y : Coordinates to be transformed to the other system
    '''
    arctan = np.arctan2(np.cos(y0)*np.sin(y)-np.sin(y0)*np.cos(y)*np.cos(x-x0),
                        np.cos(y)*np.sin(x-x0))

    a = a0 + arctan
    b = np.arcsin(np.sin(y0)*np.sin(y)+np.cos(y0)*np.cos(y)*np.cos(x-x0))

    # Force longitude to be [0,2pi]
    a[a < 0] += 2*np.pi

    return a, b

def equatorial_to_dipole(ra, dec, dipole_ra, dipole_dec):
    '''Transform equatorial coordinates to a coordinate system with dipole at the pole'''
    dipole_dec = np.deg2rad(dipole_dec)
    dipole_ra = np.deg2rad(dipole_ra)
    dipole_lon = np.deg2rad(0.)

    lon, lat = transform_spherical_coordinate_system(dipole_ra, dipole_dec, 
                                                     dipole_lon, ra, dec)

    return lon, lat

def dipole_to_equatorial(lon, lat, dipole_ra, dipole_dec):
    '''Transform back from dipole to equatorial coordinates'''
    dipole_dec = np.deg2rad(dipole_dec)
    dipole_ra = np.deg2rad(dipole_ra-90.)
    dipole_lon = np.deg2rad(90.)

    ra, dec = transform_spherical_coordinate_system(dipole_lon, dipole_dec, 
                                                    dipole_ra, lon, lat)

    return ra, dec

def equatorial_to_galactic(ra,dec):
    '''Transform from equatorial to galactic coordinates'''
    dec_ngp = np.deg2rad(27.13) #degrees
    ra_ngp = np.deg2rad(192.85) #degrees
    l_ncp = np.deg2rad(122.93-90.) # degrees

    l, b = transform_spherical_coordinate_system(ra_ngp, dec_ngp, l_ncp, ra, dec)

    return l,b

def galactic_to_equatorial(l, b):
    '''Transform from galactic to equatorial coordinates'''
    dec_ngp = np.deg2rad(27.13) #degrees
    ra_ngp = np.deg2rad(192.85-90.) #degrees
    l_ncp = np.deg2rad(122.93) #degrees

    ra, dec = transform_spherical_coordinate_system(l_ncp, dec_ngp, ra_ngp, l, b)

    return ra, dec

def angle_dipole(angle_rest, beta):
    '''Change observed angles given dipole'''
    angle_out = np.arctan2(np.sin(angle_rest)*np.sqrt(1-beta**2),beta+np.cos(angle_rest))
    return angle_out

def flux_dipole(beta, alpha, theta):
    '''Change observed flux densities given dipole and spectral index'''
    return ((1+beta*np.cos(theta))/np.sqrt(1-beta**2))**(1+alpha)

def make_dipole_healpix(NSIDE, a, x, y=None, z=None):
    '''
    Probability density function of a dipole. 
    Returns 1 + a * cos(theta) for all pixels in hp.nside2npix(nside)
    Function adapted from astrotools.healpytools
    (https://astro.pages.rwth-aachen.de/astrotools/)

    Keyword arguments:
    NSIDE (int) -- Resolution of the healpix map
    a (float) -- amplitude of the dipole
    x (array) -- x-coordinate of the center or
                 numpy array with center coordinates (cartesian definition)
    y (array) -- y-coordinate of the center
    z (array) -- z-coordinate of the center
    '''
    if y is None and z is None:
        direction = np.array(x, dtype=np.float)
    else:
        direction = np.array([x, y, z], dtype=np.float)

    # normalize to one
    direction /= np.sqrt(np.sum(direction ** 2))
    npix = hp.nside2npix(NSIDE)
    v = np.array(hp.pix2vec(NSIDE, np.arange(npix)))

    cos_angle = np.sum(v.T * direction, axis=1)

    return 1 + a * cos_angle

def make_dipole_discrete(theta, phi, a, x, y=None, z=None, angle_only=False):
    '''
    Probability density function of a dipole. Returns 1 + a * cos(theta)
    Function adapted from astrotools.healpytools 
    (https://astro.pages.rwth-aachen.de/astrotools/)

    Keyword arguments:
    theta, phi (array) -- Coordinates of points
    a (float) -- amplitude of the dipole
    x (array) -- x-coordinate of the center or 
                 numpy array with center coordinates (cartesian definition)
    y (array) -- y-coordinate of the center
    z (array) -- z-coordinate of the center
    '''
    # Convert to vector
    v = hp.ang2vec(theta, phi)
    if y is None and z is None:
        direction = np.array(x, dtype=np.float)
    else:
        direction = np.array([x, y, z], dtype=np.float)

    # normalize to one
    direction /= np.sqrt(np.sum(direction ** 2))
    cos_angle = np.sum(v * direction, axis=1)

    if angle_only:
        return np.arccos(cos_angle)
    else:
        return 1 + a * cos_angle

def plot_sampler(result, truths, outdir, label, plot_chain=False):
    '''
    Plot MCMC results

    Keyword arguments:
    sampler -- emcee sampler instance
    sim_amp (float) -- True value of dipole amplitude
    dir_theta, dir_phi (float) -- True direction dipole
    '''
    samples = result.samples
    samples[:,1], samples[:,2] = THETAPHItoRADEC(samples[:,1], samples[:,2])
    labels = ["amp", "RA", "DEC"]

    samples = samples[100:]
    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12},
                       title_fmt='.3g', truths=truths,
                       bins=50, smooth=0.9, color='navy',
                       plot_density=False, plot_datapoints=True, fill_contours=True)
    plt.savefig(os.path.join(outdir,label+'_corner.png'), dpi=300)
    plt.close()

    if plot_chain:
        fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
        for i in range(3):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.savefig(os.path.join(outdir,label+'_chain.png'), dpi=300)
        plt.close()

    for i in range(3):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f'Maximum likelihood for {labels[i]} at {mcmc[1]:.2e} -{q[0]:.2e} + {q[1]:.2e}')

    dipole_amp = np.percentile(samples[:, 0], [16, 50, 84])
    return dipole_amp