import bilby
import numpy as np
import healpy as hp

from scipy.special import gammaln

from . import helpers

def lnl_poisson(obs, model, parameters):
    """ Return log likelihood of poisson distribution """
    mean_counts = parameters['monopole']

    lnl = ( - np.sum(mean_counts*model) 
            + np.sum(obs*np.log(mean_counts*model))
            - np.sum(gammaln(obs+1)) )
    return lnl

def lnl_negativebinomial(obs, model, parameters):
    """ Return log likelihood of negative binomial distribution """
    mean_counts = parameters['monopole']
    p = parameters['p']

    r = mean_counts * p / (1-p)
    lnl = ( + np.sum(gammaln(obs+r*model))
            - np.sum(gammaln(r*model))
            - np.sum(gammaln(obs+1))
            + np.log(1-p) * np.sum(obs)
            + np.log(p) * np.sum(r*model))
    return lnl

class DipoleLikelihood(bilby.Likelihood):

    def __init__(self, obs, cells, stat, nside=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction

        Keyword arguments:
        obs (array)   -- Observations of number counts per pixel
        cells (array) -- Pixels indices corresponding to the measurements
        nside (int)   -- Resolution of the healpix map
        """
        super().__init__(parameters={'amp': None, 'dipole_ra': None, 
                                     'dipole_dec': None, 'monopole':None,
                                     'p':None})
        self.obs = obs

        # Determine which likelihood to use
        if stat == 'poisson':
            self.lnl_func = lnl_poisson
        if stat == 'nb':
            self.lnl_func = lnl_negativebinomial

        # Get pixel directions
        if nside is None:
            self.theta, self.phi = cells
        else:
            npix = hp.nside2npix(nside)
            theta, phi = hp.pix2ang(nside, np.arange(npix))
            self.theta, self.phi = theta[~cells], phi[~cells]

    def log_likelihood(self):
        amplitude = self.parameters['amp']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']

        # Create model dipole
        dipole_theta, dipole_phi = helpers.RADECtoTHETAPHI(dipole_ra, dipole_dec)
        dipole_direction = hp.ang2vec(dipole_theta, dipole_phi)

        dipole = helpers.make_dipole(self.theta, self.phi, 
                                    amplitude, dipole_direction)

        lnl = self.lnl_func(self.obs, dipole, self.parameters)
        return lnl

class DoubleDipoleLikelihood(bilby.Likelihood):

    def __init__(self, obs, cells, stat, nside=None):
        """
        Poisson likelihood of two dipoles with some amplitude and direction

        Keyword arguments:
        obs (array)   -- Observations of number counts per pixel
        cells (array) -- Pixels indices corresponding to the measurements
        nside (int)   -- Resolution of the healpix map
        """
        super().__init__(parameters={'amp': None, 'dipole_ra': None, 'dipole_dec': None,
                                     'amp2': None, 'dipole2_ra': None, 'dipole2_dec': None,
                                     'monopole':None,'p':None})
        self.obs = obs

        # Determine which likelihood to use
        if stat == 'poisson':
            self.lnl_func = lnl_poisson
        if stat == 'nb':
            self.lnl_func = lnl_negativebinomial

        # Get pixel directions
        if nside is None:
            self.theta, self.phi = cells
        else:
            npix = hp.nside2npix(nside)
            theta, phi = hp.pix2ang(nside, np.arange(npix))
            self.theta, self.phi = theta[~cells], phi[~cells]

    def log_likelihood(self):
        amplitude = self.parameters['amp']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']

        amplitude2 = self.parameters['amp2']
        dipole2_ra = self.parameters['dipole2_ra']
        dipole2_dec = self.parameters['dipole2_dec']

        # Create model dipole
        dipole_theta, dipole_phi = helpers.RADECtoTHETAPHI(dipole_ra, dipole_dec)
        dipole_direction = hp.ang2vec(dipole_theta, dipole_phi)

        # Create second dipole
        dipole2_theta, dipole2_phi = helpers.RADECtoTHETAPHI(dipole2_ra, dipole2_dec)
        dipole2_direction = hp.ang2vec(dipole2_theta, dipole2_phi)

        dipole       = helpers.make_dipole(self.theta, self.phi,
                                           amplitude, dipole_direction)
        extra_dipole = helpers.make_dipole(self.theta, self.phi,
                                           amplitude, dipole_direction)
        model = dipole + extra_dipole

        lnl = self.lnl_func(self.obs, model, self.parameters)
        return lnl

class RMSLikelihood(bilby.Likelihood):

    def __init__(self, obs_counts, obs_rms, cells, sigma_ref, stat, nside=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction,
        additionally modeling a power law relation between local noise and counts

        Keyword arguments:
        obs_counts (array) -- Observations of number of galaxies per pixel
        obs_rms (array)    -- Observations of noise per pixel
        cells (array)      -- Pixels corresponding to the measurements
        sigma_ref (float)  -- RMS reference value
        nside (int)        -- Resolution of the healpix map
        """
        super().__init__(parameters={'amp': None, 'dipole_ra': None, 'dipole_dec': None,
                                     'monopole': None, 'x': None, 'p':None})
        self.obs_counts = obs_counts
        self.obs_rms = obs_rms
        self.sigma_ref = sigma_ref

        # Determine which likelihood to use
        if stat == 'poisson':
            self.lnl_func = lnl_poisson
        if stat == 'nb':
            self.lnl_func = lnl_negativebinomial

        # Get pixel directions
        if nside is None:
            self.theta, self.phi = cells
        else:
            npix = hp.nside2npix(nside)
            theta, phi = hp.pix2ang(nside, np.arange(npix))
            self.theta, self.phi = theta[~cells], phi[~cells]

    def log_likelihood(self):
        amplitude = self.parameters['amp']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']
        x = self.parameters['x']

        # Create model dipole
        dipole_theta, dipole_phi = helpers.RADECtoTHETAPHI(dipole_ra, dipole_dec)
        dipole_direction = hp.ang2vec(dipole_theta, dipole_phi)

        dipole = helpers.make_dipole(self.theta, self.phi, 
                                    amplitude, dipole_direction)

        # Mean counts as power law
        model = dipole*(self.obs_rms/self.sigma_ref)**(-x)
        lnl = self.lnl_func(self.obs, model, self.parameters)
        return lnl

class LinearLikelihood(bilby.Likelihood):

    def __init__(self, obs_counts, obs_lin, cells, stat, nside=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction,
        additionally modeling a linear relation between counts and some defined observable

        Keyword arguments:
        obs_counts (array) -- Observations of number of galaxies per pixel
        obs_lin (array)    -- Observations of chosen observable per pixel
        cells (array)      -- Pixels corresponding to the measurements
        nside (int)        -- Resolution of the healpix map
        """
        super().__init__(parameters={'amp': None, 'dipole_ra': None, 'dipole_dec': None,
                                     'monopole':None, 'lin_amp': None, 'p':None})
        self.obs_counts = obs_counts
        self.obs_lin = obs_lin

        # Determine which likelihood to use
        if stat == 'poisson':
            self.lnl_func = lnl_poisson
        if stat == 'nb':
            self.lnl_func = lnl_negativebinomial

        # Get pixel directions
        if nside is None:
            self.theta, self.phi = cells
        else:
            npix = hp.nside2npix(nside)
            theta, phi = hp.pix2ang(nside, np.arange(npix))
            self.theta, self.phi = theta[~cells], phi[~cells]

    def log_likelihood(self):
        amplitude = self.parameters['amp']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']

        mean_counts = self.parameters['monopole']
        lin_amp = self.parameters['lin_amp']

        # Create model dipole
        dipole_theta, dipole_phi = helpers.RADECtoTHETAPHI(dipole_ra, dipole_dec)
        dipole_direction = hp.ang2vec(dipole_theta, dipole_phi)

        dipole = helpers.make_dipole_discrete(self.theta, self.phi,
                                              amplitude, dipole_direction)

        # Mean counts as absolute cosine
        model = dipole * (1 - lin_amp * self.obs_ang)
        lnl = self.lnl_func(self.obs, model, self.parameters)
        return lnl

class MultiLikelihood(bilby.Likelihood):

    def __init__(self, all_obs, all_cells, stat, nsides=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction
        for multiple catalogues

        Keyword arguments:
        all_obs (list of arrays)   -- Observations of number of galaxies per 
                                      healpix pixel, per catalog
        all_cells (list of arrays) -- Pixels indices per catalog
        nsides (list of ints)      -- Resolutions of the healpix maps
        """
        parameters = {'amp': None, 'dipole_ra': None, 'dipole_dec': None}
        for i in range(len(all_obs)):
            parameters[f'monopole_{i}'] = None
            parameters[f'p_{i}'] = None

        super().__init__(parameters)
        self.all_obs = all_obs

        # Determine which likelihood to use
        if stat == 'poisson':
            self.lnl_func = lnl_poisson
        if stat == 'nb':
            self.lnl_func = lnl_negativebinomial

        # Get pixel directions
        self.theta, self.phi = [], []
        for i, nside in enumerate(nsides):
            if nside is None:
                theta, phi = all_cells[i]
                self.theta.append(theta)
                self.phi.append(phi)
            else:
                npix = hp.nside2npix(nside)
                theta, phi = hp.pix2ang(nside, np.arange(npix))
                self.theta.append(theta[~all_cells[i]])
                self.phi.append(phi[~all_cells[i]])

    def log_likelihood(self):
        amplitude = self.parameters['amp']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']

        # Create model dipole
        dipole_theta, dipole_phi = helpers.RADECtoTHETAPHI(dipole_ra, dipole_dec)
        dipole_direction = hp.ang2vec(dipole_theta, dipole_phi)

        # Calculate log likelihood for different datasets
        lnl = 0
        for i, obs in enumerate(self.all_obs):

            dipole_model = helpers.make_dipole(self.theta[i],self.phi[i],
                                               amplitude, dipole_direction)

            nparams = {}
            nparams['monopole'] = self.parameters[f'monopole_{i}']
            nparams['p'] = self.parameters[f'p_{i}']
            lnl += self.lnl_func(obs, dipole_model, nparams)

        return lnl

class MultiVelocityLikelihood(bilby.Likelihood):

    def __init__(self, all_obs, all_cells, alpha, x, stat, nsides=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction
        for multiple catalogues. Amplitude is used to derive velocity.

        Keyword arguments:
        all_obs (list of arrays)   -- Observations of number of galaxies per 
                                      healpix pixel, per catalog
        all_cells (list of arrays) -- Pixels indices per catalog
        nsides (list of ints)      -- Resolutions of the healpix maps
        """
        parameters = {'beta': None, 'dipole_ra': None, 'dipole_dec': None}
        for i in range(len(all_obs)):
            parameters[f'monopole_{i}'] = None
            parameters[f'p_{i}'] = None

        super().__init__(parameters)
        self.all_obs = all_obs
        self.alpha = alpha
        self.x = x

        # Determine which likelihood to use
        if stat == 'poisson':
            self.lnl_func = lnl_poisson
        if stat == 'nb':
            self.lnl_func = lnl_negativebinomial

        # Get pixel directions
        self.theta, self.phi = [], []
        for i, nside in enumerate(nsides):
            if nside is None:
                theta, phi = all_cells[i]
                self.theta.append(theta)
                self.phi.append(phi)
            else:
                npix = hp.nside2npix(nside)
                theta, phi = hp.pix2ang(nside, np.arange(npix))
                self.theta.append(theta[~all_cells[i]])
                self.phi.append(phi[~all_cells[i]])

    def log_likelihood(self):
        beta = self.parameters['beta']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']

        # Create model dipole
        dipole_theta, dipole_phi = helpers.RADECtoTHETAPHI(dipole_ra, dipole_dec)
        dipole_direction = hp.ang2vec(dipole_theta, dipole_phi)

        # Calculate log likelihood for different datasets
        lnl = 0
        for i, obs in enumerate(self.all_obs):
            amplitude = (2 + self.x[i]*(1+self.alpha[i]))*beta

            dipole_model = helpers.make_dipole(self.theta[i],self.phi[i],
                                               amplitude, dipole_direction)

            nparams = {}
            nparams['monopole'] = self.parameters[f'monopole_{i}']
            nparams['p'] = self.parameters[f'p_{i}']
            lnl += self.lnl_func(obs, dipole_model, nparams)

        return lnl

class MultiCombinedLikelihood(bilby.Likelihood):

    def __init__(self, all_obs, all_cells, alpha, x, stat, nsides=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction
        for multiple catalogues. Kinematic and residual contribution 
        to the total dipole amplitude are separated.

        Keyword arguments:
        all_obs (list of arrays)   -- Observations of number of galaxies per 
                                      healpix pixel, per catalog
        all_cells (list of arrays) -- Pixels indices per catalog
        nsides (list of ints)      -- Resolutions of the healpix maps
        """
        parameters = {'beta': None, 'amp': None, 'dipole_ra': None, 'dipole_dec': None}
        for i in range(len(all_obs)):
            parameters[f'monopole_{i}'] = None
            parameters[f'p_{i}'] = None

        super().__init__(parameters)
        self.all_obs = all_obs
        self.alpha = alpha
        self.x = x

        # Determine which likelihood to use
        if stat == 'poisson':
            self.lnl_func = lnl_poisson
        if stat == 'nb':
            self.lnl_func = lnl_negativebinomial

        # Get pixel directions
        self.theta, self.phi = [], []
        for i, nside in enumerate(nsides):
            if nside is None:
                theta, phi = all_cells[i]
                self.theta.append(theta)
                self.phi.append(phi)
            else:
                npix = hp.nside2npix(nside)
                theta, phi = hp.pix2ang(nside, np.arange(npix))
                self.theta.append(theta[~all_cells[i]])
                self.phi.append(phi[~all_cells[i]])

    def log_likelihood(self):
        beta = self.parameters['beta']
        amp = self.parameters['amp']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']

        # Create model dipole
        dipole_theta, dipole_phi = helpers.RADECtoTHETAPHI(dipole_ra, dipole_dec)
        dipole_direction = hp.ang2vec(dipole_theta, dipole_phi)

        # Calculate log likelihood for different datasets
        lnl = 0
        for i, obs in enumerate(self.all_obs):
            full_amplitude = (2 + self.x[i]*(1+self.alpha[i]))*beta + amp

            dipole_model = helpers.make_dipole(self.theta[i],self.phi[i],
                                               amplitude, dipole_direction)

            nparams = {}
            nparams['monopole'] = self.parameters[f'monopole_{i}']
            nparams['p'] = self.parameters[f'p_{i}']
            lnl += self.lnl_func(obs, dipole_model, nparams)

        return lnl

class MultiCombinedDirLikelihood(bilby.Likelihood):

    def __init__(self, all_obs, all_cells, alpha, x, stat, nsides=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction
        for multiple catalogues. Kinematic and residual contribution 
        to the total dipole amplitude are separated, these contributions
        can also have different directions.

        Keyword arguments:
        all_obs (list of arrays)   -- Observations of number of galaxies per 
                                      healpix pixel, per catalog
        all_cells (list of arrays) -- Pixels indices per catalog
        nsides (list of ints)      -- Resolutions of the healpix maps
        """
        parameters = {'beta': None, 'vel_ra': None, 'vel_dec': None,
                      'amp': None, 'dipole_ra': None, 'dipole_dec': None}
        for i in range(len(all_obs)):
            parameters[f'monopole_{i}'] = None
            parameters[f'p_{i}'] = None

        super().__init__(parameters)
        self.all_obs = all_obs
        self.alpha = alpha
        self.x = x

        # Determine which likelihood to use
        if stat == 'poisson':
            self.lnl_func = lnl_poisson
        if stat == 'nb':
            self.lnl_func = lnl_negativebinomial

        # Get pixel directions
        self.theta, self.phi = [], []
        for i, nside in enumerate(nsides):
            if nside is None:
                theta, phi = all_cells[i]
                self.theta.append(theta)
                self.phi.append(phi)
            else:
                npix = hp.nside2npix(nside)
                theta, phi = hp.pix2ang(nside, np.arange(npix))
                self.theta.append(theta[~all_cells[i]])
                self.phi.append(phi[~all_cells[i]])

    def log_likelihood(self):
        beta = self.parameters['beta']
        vel_ra = self.parameters['vel_ra']
        vel_dec = self.parameters['vel_dec']

        amp = self.parameters['amp']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']

        # Create velocity dipole
        vel_theta, vel_phi = helpers.RADECtoTHETAPHI(vel_ra, vel_dec)
        velocity_direction = hp.ang2vec(vel_theta, vel_phi)

        # Create intrinsic dipole
        dipole_theta, dipole_phi = helpers.RADECtoTHETAPHI(dipole_ra, dipole_dec)
        dipole_direction = hp.ang2vec(dipole_theta, dipole_phi)

        # Calculate log likelihood for different datasets
        lnl = 0
        for i, obs in enumerate(self.all_obs):
            kin_amp = (2 + self.x[i]*(1+self.alpha[i]))*beta

            kin_dipole = helpers.make_dipole(self.theta[i],self.phi[i],
                                             kin_amp, velocity_direction)
            int_dipole = helpers.make_dipole(self.theta[i],self.phi[i],
                                             amp, dipole_direction)
            dipole_model = kin_dipole + int_dipole - 1

            nparams = {}
            nparams['monopole'] = self.parameters[f'monopole_{i}']
            nparams['p'] = self.parameters[f'p_{i}']
            lnl += self.lnl_func(obs, dipole_model, nparams)

        return lnl