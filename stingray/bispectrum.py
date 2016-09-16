from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.fftpack
import stingray.lightcurve as lightcurve

__all__ = ["Bispectrum", "bicoherence"]


def bicoherence(lc):
    """
    Estimates Bicoherence  of a light curve.

    Parameters
    ----------
    lc: lightcurve.Lightcurve object


    Returns
    -------
    coh : np.ndarray
        Bicoherence array
    """

    if not isinstance(lc, lightcurve.Lightcurve):
        raise TypeError("lc must be a lightcurve.Lightcurve object")

    bispec = Bispectrum(lc)

    return bispec.bicoherence()


class Bispectrum(object):

    def __init__(self, lc=None, n=64, nfft=128):
        """
        Make a  Bispectrum from a light curve.
        You can also make an empty Bispectrum object to populate with your
        own fourier-transformed data (this can sometimes be useful when making
        binned periodograms).

        Parameters
        ----------
        lc: lightcurve.Lightcurve object, optional, default None
            The light curve data to be Fourier-transformed.

        n : int
            Number of samples in each segment Default :64(expecting huge data)

        nfft : int
            Fft size Default is 128


        Attributes
        ----------
        m: int
            Number of segments

        n : int
            Number of samples in each segment Default :64(expecting huge data)

        nfft : int
            Fft size Default is 128

        ncounts  : int
            total number of samples from the input signal

        bispectrum : numpy.ndarray
            the array of Bispectrum

        """

        if lc is not None:
            pass
        else:
            self.m = None
            self.nfft = nfft
            self.n = n
            self.ncounts = None
            return
        self.n = n
        self.nfft = nfft

        if not isinstance(lc, lightcurve.Lightcurve):
            raise TypeError("lc must be a lightcurve.Lightcurve object")

        self.lc = lc

        self.ncounts = self.lc.counts.shape[0]

        if isinstance(n, int) is False:
            raise TypeError("n(No of samples in segment) must be an integer")

        if isinstance(nfft, int) is False:
            raise TypeError("nfft must be an integer")

        self.m = self.ncounts // self.n

        self.bispectrum = self._compute_bispectrum()

    def _compute_bispectrum(self):
        """

        Computes the bispectrum of lightcurve from the given Bispectrum
        object

        B(f1,f2) = Xf1 * Xf2 * conjugate(X(f1+f2))

        where Xf1 is the magnitude of fft at frequency f1
        where Xf2 is the magnitude of fft at frequency f2
        where X(f1+f2) is the magnitude of fft at frequency f1+f2


        References
        ----------
        .. [1] Bispectrum estimation: A digital signal processing framework
            C.L. Nikias ; M.R. Raghuveer   DOI: 10.1109/PROC.1987.13824

        """
        bispec = np.zeros([self.nfft, self.nfft], dtype=complex)

        hankelmatrix = hankel(np.arange(self.nfft),
                              np.array([self.nfft - 1] + list(range(self.nfft - 1))))

        # hankel matrix when used for indexing repective arrays helps
        # us in obtaining summed frequencies X(f1+f2)

        pseg = np.arange(self.n).transpose()

        # we iterate over all segments , summing the fft over those iterations
        # finally divide by number of segments to obtain mean
        # followed by returning bispectrum

        for i in range(self.m):
            iseg = self.lc.counts[pseg].reshape(1, -1)
            Xf = scipy.fftpack.fft(iseg - np.mean(iseg), self.nfft) / self.n
            CXf = np.conjugate(Xf).ravel(order='F')
            bispec = bispec + (Xf * np.transpose(Xf)) * \
                CXf[hankelmatrix].reshape(self.nfft, self.nfft)
            pseg = pseg + self.n

        # Shift the zero-frequency component to the center of the spectrum.
        bispec = scipy.fftpack.fftshift(bispec) / self.m
        return bispec

    def bicoherence(self):
        """
        Computes the bicoherence of lightcurve from the given Bispectrum
        object

        B(f1,f2) = abs(bispectrum(f1,f2))**2 / Pf1 * Pf2 * P(f1+f2)

        where Pf1 is the magnitude of powerspectra at frequency f1
        where Pf2 is the magnitude of powerspectra at frequency f2
        where P(f1+f2) is the magnitude of powerspectra  at frequency f1+f2


         References
        ----------
        .. [1] Bispectrum estimation: A digital signal processing framework
                    C.L. Nikias ; M.R. Raghuveer   DOI: 10.1109/PROC.1987.13824

        """

        bicoh = np.zeros([self.nfft, self.nfft], dtype=complex)
        hankelmatrix = hankel(np.arange(self.nfft),
                              np.array([self.nfft - 1] + list(range(self.nfft - 1))))

        # hankel matrix when used for indexing repective arrays helps
        # us in obtaining summed frequencies X(f1+f2)

        pseg = np.arange(self.n).transpose()

        powerspec = np.zeros([1, self.nfft], dtype=complex)

        # we iterate over all segments ,summing the power over those iterations
        # finally divide by number of segments to obtain mean
        # followed by returning bicoherence
        for i in range(self.m):
            iseg = self.lc.counts[pseg].reshape(1, -1)
            Xf = scipy.fftpack.fft(iseg - np.mean(iseg), self.nfft) / self.n
            CXf = np.conjugate(Xf)
            powerspec = powerspec + (Xf * CXf)
            pseg = pseg + self.n

        powerspec = powerspec.ravel(order='F')

        powerspec = scipy.fftpack.fftshift(powerspec) / self.m

        bicoh = abs(self.bispectrum)**2 / \
            (powerspec * np.transpose(powerspec) *
             powerspec[hankelmatrix].reshape(self.nfft, self.nfft))

        return bicoh
