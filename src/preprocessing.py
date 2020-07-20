from __future__ import division, print_function
import numpy as np


def unit_variance(spectra):
    """This function transforms spectra using unit variance method

    x^uv_{i, j} = x_{i, j} / std(x_j)
    i is the spectra index, j is the wavelength index

    Parameters
    -----------
    spectra: ndarray, row vectors are not suited
        contains spectra infomation, with i spectrums and j wavelengths
    Returns
    ----------
    uv_spec: ndarray
        spectra after unit variance transform
    """
    std = np.std(spectra, ddof=1, axis=0)
    std = std.reshape(1, std.shape[0])
    uv_spec = spectra / std
    return uv_spec


def mean_center(spectra):
    """This function transforms spectra using mean centering method

    x^mc_{i, j} = x_{i, j} - mean(x_j)
    i is the spectra index, j is the wavelength index

    Parameters
    ------------
    spectra: ndarray
        contains spectra infomation, with i spectrums and j wavelengths

    Returns
    ----------
    mc_spectra: ndarray
        spectra after mean centering

    """
    mean = np.mean(spectra, axis=0)
    mc_spectra = spectra - mean * np.ones(mean.shape)
    return mc_spectra


def autoscale(spectra):
    """This function transforms spectra using autoscale

    autoscale applies mean centering and unit variance both

    Parameters
    -------------
    spectra: ndarray
        contains spectra infomation, with i spectrums and j wavelengths

    Returns
    -------------
    as_spectra: ndarray
        spectra after mean centering
    """
    as_spectra = mean_center(spectra)
    as_spectra = unit_variance(as_spectra)
    return as_spectra


def UVN(spectra):
    """This function transforms spectra using UVN (from SIMCA)

    Same as autoscale, except that the mean is replaced by 0
    wUVN = sqrt(sum(X_j ** 2) / len(X_j))
    X^UVN_{i, j} = X_{i, j} / wUVN
    Parameters
    -------------
    spectra: ndarray
        contains spectra infomation, with i spectrums and j wavelengths

    Returns
    -------------
    uvn_spectra: ndarray
        spectra after mean centering
    """

    wUVN = np.sqrt(np.sum(spectra ** 2, axis=0) / spectra.shape[0])
    wUVN = wUVN.reshape(1, spectra.shape[1])
    uvn_spectra = spectra / wUVN
    return uvn_spectra


def SNV(spectra):
    """ This function transforms spectra using Standard Normal Variates method

    x^SNV_{i, j} = (x{i, j} - mean(X_i) / s_i);
    i is the spectra index, j is the wavelength index

    Reference
    --------
    [Automated system for the on-line monitoring of powder blending processes\
    using near-infrared spectroscopy Part II. Qualitative approaches to blend\
    evaluation](http://www.sciencedirect.com/science/article/pii/S0731708598000259?via%3Dihub)

    Parameters
    ---------
    spectra: ndarray
        contains spectrum infomation, with i spectrums and j wavelengths

    Returns
    ----------
    snv_spectra: ndarray
        spectra after SNV transform
    """

    mean = np.mean(spectra, axis=1)
    mean = mean.reshape(mean.shape[0], 1)
    std = np.std(spectra, ddof=1, axis=1)
    std = std.reshape(std.shape[0], 1)
    snv_spectra = (spectra - mean) / std
    return snv_spectra


def MSC(spectra, reference=None, extend_order=0):
    """This function transforms spectra using (extend) multiplicative scattering correction

    MSC corrects the multiplicative effects caused by different pathlengths and scattering effects.
    Change extend_order to a non-zero number to get EMSC

    Reference
    -----------

    Parameters
    ----------
    spectra : [type]
        [description]
    reference : [type], optional
        [description] (the default is None, which [default_description])
    extend_order : int, optional
        [description] (the default is 2, which [default_description])

    """

    if spectra.shape[0] == 1:
        spectra = spectra.reshape(1, -1)
    if reference is None:
        reference = np.mean(spectra, axis=0)


