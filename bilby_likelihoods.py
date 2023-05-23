import bilby
import numpy as np
import healpy as hp

from scipy.special import gammaln

import helpers

class PoissonLikelihood(bilby.Likelihood):

    def __init__(self, obs, cells, NSIDE=None):
        '''
        Poisson likelihood of a dipole with some amplitude and direction

        Keyword arguments:
        obs (array)   -- Observations of number counts per healpix pixel
        cells (array) -- Pixels indices corresponding to the measurements
        NSIDE (int)   -- Resolution of the healpix map
        '''
        super().__init__(parameters={'amp': None, 'dipole_ra': None, 
                                     'dipole_dec': None, 'lambda':None})
        self.obs = obs
        self.NSIDE = NSIDE
        self.cells = cells

    def log_likelihood(self):
        amplitude = self.parameters['amp']
        dipole_ra = self.parameters['dipole_ra']
        dipole_dec = self.parameters['dipole_dec']
        mean_counts = self.parameters['lambda']

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

class PoissonRMSLikelihood(bilby.Likelihood):

    def __init__(self, obs_counts, obs_rms, cells, sigma_ref, NSIDE=None):
        '''
        Poisson likelihood of a dipole with some amplitude and direction,
        modeling a power law relation between local noise and counts

        Keyword arguments:
        obs_counts (array) -- Observations of number of galaxies per healpix pixel
        obs_rms (array)    -- Observations of noise per healpix pixel
        cells (array)      -- Pixels indices corresponding to the measurements
        sigma_ref (float)  -- RMS reference value
        NSIDE (int)        -- Resolution of the healpix map
        '''
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

class MultiPoissonLikelihood(bilby.Likelihood):

    def __init__(self, all_obs, all_cells, NSIDES=None):
        '''
        Poisson likelihood of a dipole with some amplitude and direction
        for multiple catalogues

        Keyword arguments:
        all_obs (list of arrays)   -- Observations of number of galaxies per 
                                      healpix pixel, per catalog
        all_cells (list of arrays) -- Pixels indices per catalog
        NSIDEs (list of ints)      -- Resolutions of the healpix maps
        '''
        parameters = {'amp': None, 'dipole_ra': None, 'dipole_dec': None}
        for i in range(len(all_obs)):
            parameters[f'lambda_{i}'] = None

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
                pointings_theta, pointings_phi = all_cells[i]
                dipole_model = helpers.make_dipole_discrete(pointings_theta,
                                                            pointings_phi,
                                                            amplitude,
                                                            dipole_direction)
            else:
                dipole = helpers.make_dipole_healpix(self.NSIDES[i],
                                                     amplitude,
                                                     dipole_direction)
                dipole_model = dipole[~self.all_cells[i]]

            mean_counts = self.parameters[f'lambda_{i}']
            lnl  +=  (- np.sum(mean_counts*dipole_model) 
                      + np.sum(obs*np.log(mean_counts*dipole_model)) 
                      - np.sum(gammaln(obs+1)))

        return lnl