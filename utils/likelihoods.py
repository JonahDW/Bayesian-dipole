import bilby
import numpy as np
import healpy as hp

from scipy.special import gammaln

from . import helpers

class PoissonLikelihood(bilby.Likelihood):

    def __init__(self, obs, cells, NSIDE=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction

        Keyword arguments:
        obs (array)   -- Observations of number counts per pixel
        cells (array) -- Pixels indices corresponding to the measurements
        NSIDE (int)   -- Resolution of the healpix map
        """
        super().__init__(parameters={'amp': None, 'dipole_ra': None, 
                                     'dipole_dec': None, 'monopole':None})
        self.obs = obs
        self.NSIDE = NSIDE
        self.cells = cells

    def log_likelihood(self):
        amplitude = self.parameters['amp']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']
        mean_counts = self.parameters['monopole']

        # Create model dipole
        dipole_theta, dipole_phi = helpers.RADECtoTHETAPHI(dipole_ra, dipole_dec)
        dipole_direction = hp.ang2vec(dipole_theta, dipole_phi)

        if self.NSIDE is None:
            pointings_theta, pointings_phi = self.cells
            dipole = helpers.make_dipole_discrete(pointings_theta, pointings_phi, 
                                                  amplitude, dipole_direction)
        else:
            dipole = helpers.make_dipole_healpix(self.NSIDE, amplitude, dipole_direction)
            dipole = dipole[~self.cells]

        return ( - np.sum(mean_counts*dipole) 
                 + np.sum(self.obs*np.log(mean_counts*dipole))
                 - np.sum(gammaln(self.obs+1)) )


class PoissonDoubleDipoleLikelihood(bilby.Likelihood):

    def __init__(self, obs_counts, cells, NSIDE=None):
        """
        Poisson likelihood of two dipoles with some amplitude and direction

        Keyword arguments:
        obs (array)   -- Observations of number counts per pixel
        cells (array) -- Pixels indices corresponding to the measurements
        NSIDE (int)   -- Resolution of the healpix map
        """
        super().__init__(parameters={'amp': None, 'dipole_ra': None, 'dipole_dec': None,
                                     'amp2': None, 'dipole2_ra': None, 'dipole2_dec': None,
                                     'monopole':None,})
        self.obs_counts = obs_counts
        self.NSIDE = NSIDE
        self.cells = cells

    def log_likelihood(self):
        mean_counts = self.parameters['monopole']
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

        if self.NSIDE is None:
            pointings_theta, pointings_phi = self.cells
            dipole = helpers.make_dipole_discrete(pointings_theta, pointings_phi,
                                                  amplitude, dipole_direction)

            extra_dipole = helpers.make_dipole_discrete(pointings_theta, pointings_phi,
                                                        amplitude, dipole_direction)
        else:
            dipole = helpers.make_dipole_healpix(self.NSIDE, amplitude, dipole_direction)
            dipole = dipole[~self.cells]

            extra_dipole = helpers.make_dipole_healpix(self.NSIDE, amplitude2, dipole2_direction)
            extra_dipole = extra_dipole[~self.cells]

        model = dipole + extra_dipole
        return (- np.sum(mean_counts*model) 
                + np.sum(self.obs_counts*np.log(mean_counts*model))
                - np.sum(gammaln(self.obs_counts+1)) )

class PoissonRMSLikelihood(bilby.Likelihood):

    def __init__(self, obs_counts, obs_rms, cells, sigma_ref, NSIDE=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction,
        additionally modeling a power law relation between local noise and counts

        Keyword arguments:
        obs_counts (array) -- Observations of number of galaxies per pixel
        obs_rms (array)    -- Observations of noise per pixel
        cells (array)      -- Pixels corresponding to the measurements
        sigma_ref (float)  -- RMS reference value
        NSIDE (int)        -- Resolution of the healpix map
        """
        super().__init__(parameters={'amp': None, 'dipole_ra': None, 'dipole_dec': None,
                                     'rms_amp': None, 'rms_pow': None})
        self.obs_counts = obs_counts
        self.obs_rms = obs_rms
        self.NSIDE = NSIDE
        self.cells = cells
        self.sigma_ref = sigma_ref

    def log_likelihood(self):
        amplitude = self.parameters['amp']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']
        rms_amplitude = self.parameters['rms_amp']
        rms_power = self.parameters['rms_pow']

        # Create model dipole
        dipole_theta, dipole_phi = helpers.RADECtoTHETAPHI(dipole_ra, dipole_dec)
        dipole_direction = hp.ang2vec(dipole_theta, dipole_phi)

        if self.NSIDE is None:
            pointings_theta, pointings_phi = self.cells
            dipole = helpers.make_dipole_discrete(pointings_theta, pointings_phi, 
                                                  amplitude, dipole_direction)
        else:
            dipole = helpers.make_dipole_healpix(self.NSIDE, amplitude, dipole_direction)
            dipole = dipole[~self.cells]

        # Mean counts as power law
        mean_counts = rms_amplitude*(self.obs_rms/self.sigma_ref)**(-rms_power)

        return ( - np.sum(mean_counts*dipole) 
                 + np.sum(self.obs_counts*np.log(mean_counts*dipole)) 
                 - np.sum(gammaln(self.obs_counts+1)) )

class PoissonLinearLikelihood(bilby.Likelihood):

    def __init__(self, obs_counts, obs_lin, cells, NSIDE=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction,
        additionally modeling a linear relation between counts and some defined observable

        Keyword arguments:
        obs_counts (array) -- Observations of number of galaxies per pixel
        obs_lin (array)    -- Observations of chosen observable per pixel
        cells (array)      -- Pixels corresponding to the measurements
        NSIDE (int)        -- Resolution of the healpix map
        """
        super().__init__(parameters={'amp': None, 'dipole_ra': None, 'dipole_dec': None,
                                     'monopole':None, 'lin_amp': None})
        self.obs_counts = obs_counts
        self.obs_ang = obs_ang
        self.NSIDE = NSIDE
        self.cells = cells

    def log_likelihood(self):
        amplitude = self.parameters['amp']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']

        mean_counts = self.parameters['monopole']
        lin_amp = self.parameters['lin_amp']

        # Create model dipole
        dipole_theta, dipole_phi = helpers.RADECtoTHETAPHI(dipole_ra, dipole_dec)
        dipole_direction = hp.ang2vec(dipole_theta, dipole_phi)

        if self.NSIDE is None:
            pointings_theta, pointings_phi = self.cells
            dipole = helpers.make_dipole_discrete(pointings_theta, pointings_phi,
                                                  amplitude, dipole_direction)
        else:
            dipole = helpers.make_dipole_healpix(self.NSIDE, amplitude, dipole_direction)
            dipole = dipole[~self.cells]

        # Mean counts as absolute cosine
        counts = mean_counts * (1 - lin_amp * self.obs_ang)
        return ( - np.sum(counts*dipole) 
               + np.sum(self.obs_counts*np.log(counts*dipole))
               - np.sum(gammaln(self.obs_counts+1)) )

class MultiPoissonLikelihood(bilby.Likelihood):

    def __init__(self, all_obs, all_cells, NSIDES=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction
        for multiple catalogues

        Keyword arguments:
        all_obs (list of arrays)   -- Observations of number of galaxies per 
                                      healpix pixel, per catalog
        all_cells (list of arrays) -- Pixels indices per catalog
        NSIDEs (list of ints)      -- Resolutions of the healpix maps
        """
        parameters = {'amp': None, 'dipole_ra': None, 'dipole_dec': None}
        for i in range(len(all_obs)):
            parameters[f'monopole_{i}'] = None

        super().__init__(parameters)
        self.all_obs = all_obs
        self.all_cells = all_cells
        self.NSIDES = NSIDES

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

            if self.NSIDES[i] is None:
                pointings_theta, pointings_phi = self.all_cells[i]
                dipole_model = helpers.make_dipole_discrete(pointings_theta,
                                                            pointings_phi,
                                                            amplitude,
                                                            dipole_direction)
            else:
                dipole = helpers.make_dipole_healpix(self.NSIDES[i],
                                                     amplitude,
                                                     dipole_direction)
                dipole_model = dipole[~self.all_cells[i]]

            mean_counts = self.parameters[f'monopole_{i}']
            lnl  +=  (- np.sum(mean_counts*dipole_model) 
                      + np.sum(obs*np.log(mean_counts*dipole_model)) 
                      - np.sum(gammaln(obs+1)))

        return lnl

class MultiPoissonVelocityLikelihood(bilby.Likelihood):

    def __init__(self, all_obs, all_cells, alpha, x, NSIDES=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction
        for multiple catalogues. Amplitude is used to derive velocity.

        Keyword arguments:
        all_obs (list of arrays)   -- Observations of number of galaxies per 
                                      healpix pixel, per catalog
        all_cells (list of arrays) -- Pixels indices per catalog
        NSIDEs (list of ints)      -- Resolutions of the healpix maps
        """
        parameters = {'beta': None, 'dipole_ra': None, 'dipole_dec': None}
        for i in range(len(all_obs)):
            parameters[f'monopole_{i}'] = None

        super().__init__(parameters)
        self.all_obs = all_obs
        self.all_cells = all_cells
        self.alpha = alpha
        self.x = x
        self.NSIDES = NSIDES

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

            if self.NSIDES[i] is None:
                pointings_theta, pointings_phi = self.all_cells[i]
                dipole_model = helpers.make_dipole_discrete(pointings_theta,
                                                            pointings_phi,
                                                            amplitude,
                                                            dipole_direction)
            else:
                dipole = helpers.make_dipole_healpix(self.NSIDES[i],
                                                     amplitude,
                                                     dipole_direction)
                dipole_model = dipole[~self.all_cells[i]]

            mean_counts = self.parameters[f'monopole_{i}']
            lnl  +=  (- np.sum(mean_counts*dipole_model) 
                      + np.sum(obs*np.log(mean_counts*dipole_model)) 
                      - np.sum(gammaln(obs+1)))

        return lnl

class MultiPoissonCombinedLikelihood(bilby.Likelihood):

    def __init__(self, all_obs, all_cells, alpha, x, NSIDES=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction
        for multiple catalogues. Kinematic and residual contribution 
        to the total dipole amplitude are separated.

        Keyword arguments:
        all_obs (list of arrays)   -- Observations of number of galaxies per 
                                      healpix pixel, per catalog
        all_cells (list of arrays) -- Pixels indices per catalog
        NSIDEs (list of ints)      -- Resolutions of the healpix maps
        """
        parameters = {'beta': None, 'amp': None, 'dipole_ra': None, 'dipole_dec': None}
        for i in range(len(all_obs)):
            parameters[f'monopole_{i}'] = None

        super().__init__(parameters)
        self.all_obs = all_obs
        self.all_cells = all_cells
        self.alpha = alpha
        self.x = x
        self.NSIDES = NSIDES

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

            if self.NSIDES[i] is None:
                pointings_theta, pointings_phi = self.all_cells[i]
                dipole_model = helpers.make_dipole_discrete(pointings_theta,
                                                            pointings_phi,
                                                            full_amplitude,
                                                            dipole_direction)
            else:
                dipole = helpers.make_dipole_healpix(self.NSIDES[i],
                                                     full_amplitude,
                                                     dipole_direction)
                dipole_model = dipole[~self.all_cells[i]]

            mean_counts = self.parameters[f'monopole_{i}']
            lnl  +=  (- np.sum(mean_counts*dipole_model) 
                      + np.sum(obs*np.log(mean_counts*dipole_model)) 
                      - np.sum(gammaln(obs+1)))

        return lnl

class MultiPoissonCombinedDirLikelihood(bilby.Likelihood):

    def __init__(self, all_obs, all_cells, alpha, x, NSIDES=None):
        """
        Poisson likelihood of a dipole with some amplitude and direction
        for multiple catalogues. Kinematic and residual contribution 
        to the total dipole amplitude are separated, these contributions
        can also have different directions.

        Keyword arguments:
        all_obs (list of arrays)   -- Observations of number of galaxies per 
                                      healpix pixel, per catalog
        all_cells (list of arrays) -- Pixels indices per catalog
        NSIDEs (list of ints)      -- Resolutions of the healpix maps
        """
        parameters = {'beta': None, 'vel_ra': None, 'vel_dec': None,
                      'amp': None, 'dipole_ra': None, 'dipole_dec': None}
        for i in range(len(all_obs)):
            parameters[f'monopole_{i}'] = None

        super().__init__(parameters)
        self.all_obs = all_obs
        self.all_cells = all_cells
        self.alpha = alpha
        self.x = x
        self.NSIDES = NSIDES

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

            if self.NSIDES[i] is None:
                pointings_theta, pointings_phi = self.all_cells[i]
                kin_dipole = helpers.make_dipole_discrete(pointings_theta,
                                                          pointings_phi,
                                                          kin_amp, velocity_direction)
                int_dipole = helpers.make_dipole_discrete(pointings_theta,
                                                          pointings_phi,
                                                          amp, dipole_direction)
                dipole_model = kin_dipole + int_dipole - 1

            else:
                kin_dipole = helpers.make_dipole_healpix(self.NSIDES[i], kin_amp,
                                                         velocity_direction)
                int_dipole = helpers.make_dipole_healpix(self.NSIDES[i], amp,
                                                         dipole_direction)
                dipole_model = kin_dipole[~self.all_cells[i]] + int_dipole[~self.all_cells[i]] - 1

            mean_counts = self.parameters[f'monopole_{i}']
            lnl  +=  (- np.sum(mean_counts*dipole_model) 
                      + np.sum(obs*np.log(mean_counts*dipole_model)) 
                      - np.sum(gammaln(obs+1)))

        return lnl