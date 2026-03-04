import bilby
import numpy as np

class CosineDeg(bilby.core.prior.Prior):

    def __init__(self, minimum=-90., maximum=90., name=None,
                 latex_label=None, unit=None, boundary=None):
        """Cosine prior with bounds"""
        super(CosineDeg, self).__init__(minimum=minimum, maximum=maximum, name=name, 
                                latex_label=latex_label, unit=unit, boundary=boundary)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in cosine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        norm = 1 / (np.sin(self.maximum * np.pi / 180) - np.sin(self.minimum * np.pi / 180))
        arcsin = np.arcsin(val / norm + np.sin(self.minimum * np.pi / 180))
        return arcsin * 180 / np.pi

    def prob(self, val):
        """Return the prior probability of val. Defined over [-pi/2, pi/2].

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        val_rad = val * np.pi / 180
        return np.cos(val_rad) / 2 * self.is_in_prior_range(val)

    def cdf(self, val):
        val_rad = val * np.pi / 180
        _cdf = np.atleast_1d((np.sin(val_rad) - np.sin(self.minimum * np.pi / 180)) /
                             (np.sin(self.maximum * np.pi / 180) 
                            - np.sin(self.minimum * np.pi / 180)))
        _cdf[val > self.maximum] = 1
        _cdf[val < self.minimum] = 0
        return _cdf

def default_dipole_priors(dipole_mode):
    # Set up dipole parameters

    priors = dict()
    injection_parameters = dict()
    if dipole_mode.lower() == 'amplitude':
        priors['amp'] = bilby.core.prior.Uniform(0,1,'$\\mathcal{D}$')
        injection_parameters['amp'] = 4.5e-3
    if dipole_mode.lower() == 'velocity':
        priors['beta'] = bilby.core.prior.Uniform(0,0.1,'$\\beta$')
        injection_parameters['beta'] = 1.23e-3
    if dipole_mode.lower() == 'combined':
        priors['amp'] = bilby.core.prior.Uniform(0,0.5,'$\\mathcal{D}$')
        priors['beta'] = bilby.core.prior.Uniform(0,0.05,'$\\beta$')
        injection_parameters['amp'] = 4.5e-3
        injection_parameters['beta'] = 1.23e-3
    if dipole_mode.lower() == 'combined-dir':
        priors['amp'] = bilby.core.prior.Uniform(0,0.5,'$\\mathcal{D}$')
        priors['beta'] = bilby.core.prior.Uniform(0,0.05,'$\\beta$')
        priors['vel_ra'] = bilby.core.prior.Uniform(0,360.,'vel RA')
        priors['vel_dec'] = cpriors.CosineDeg(-90.,90.,'vel DEC')

        injection_parameters['amp'] = 4.5e-3
        injection_parameters['beta'] = 1.23e-3
        injection_parameters['vel_ra'] = 168.
        injection_parameters['vel_dec'] = -7.

    priors['dipole_ra'] = bilby.core.prior.Uniform(0,360.,'RA')
    priors['dipole_dec'] = CosineDeg(-90.,90.,'DEC')

    injection_parameters['dipole_ra'] = 140.
    injection_parameters['dipole_dec'] = -7.

    return priors, injection_parameters