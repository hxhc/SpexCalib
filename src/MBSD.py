import numpy as np


def MBSD(spec, window_size=10):
    """Moving Block Standard Deviation method

    This function is to calculate the MBSD of spectra


    Parameters
    -----------
    spec: ndarray
        contains spectrum infomation, with i spectrums and j wavelengths
    window_size: int (default 10)
        window_size is the size of moving block

    Returns
    ---------
    mbsd_list: ndarray
        a list contains the MBSD of the apectra

    """
    mbsd_list = np.zeros(spec.shape[0] - window_size + 1)
    for i in range(spec.shape[0] - window_size + 1):
        window = spec[i:i + window_size - 1, :]
        window_sd = window.std(axis=0, ddof=1)
        mbsd = window_sd.mean()
        mbsd_list[i] = mbsd
    return mbsd_list
