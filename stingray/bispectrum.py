from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.fftpack
import stingray.lightcurve as lightcurve

# import stingray.utils as utils
# from stingray.utils import simon

__all__ = ["Bispectrum"]


class Bispectrum(object):

    def __init__(self, lc=None, m=64, nfft=128):
        """
        Parameters
        ----------
        lc: lightcurve.Lightcurve object, optional, default None
        The light curve data to be Fourier-transformed.

        m : Number of samples in each segment Default :64(expecting huge data)

        nfft : Fft size Default is 128


        attributes
        ----------
        k: Number of segments

        m : Number of samples in each segment Default :64(expecting huge data)

        nfft : Fft size Default is 128

        nsamp  : total numberof samples from the input signal

        """

        if lc is not None:
            pass
        else:
            self.k = None
            self.nfft = nfft
            self.m = m
            self.nsamp = None
            return
        self.m = m
        self.nfft = nfft
        assert isinstance(lc, lightcurve.Lightcurve), \
            "lc must be a lightcurve.Lightcurve object!"

        lc.counts = lc.counts.ravel(order='C')
        self.nsamp = lc.counts.shape[0]

        assert isinstance(m, int), "m is not an int !"

        assert isinstance(nfft, int), "nfft is not an int !"

        self.k = self.nsamp // self.m

    def compute_bispectrum(self, lc):
        bispec = np.zeros([self.nfft, self.nfft], dtype=complex)
        mask = hankel(np.arange(self.nfft),
                      np.array([self.nfft - 1] + list(range(self.nfft - 1))))
        # hankel serves the function of summed frequncy in bispectra
        Xf = 0
        CXf = 0
        pseg = np.arange(self.m).transpose()

        for i in range(self.k):
            iseg = lc.counts[pseg].reshape(1, -1)
            Xf = scipy.fftpack.fft(iseg - np.mean(iseg), self.nfft) / self.m
            CXf = np.conjugate(Xf).ravel(order='C')
            bispec = bispec + (Xf * np.transpose(Xf)) * \
                CXf[mask].reshape(self.nfft, self.nfft)
            pseg = pseg + self.m

        bispec = scipy.fftpack.fftshift(bispec) / self.k
        return bispec

    def bicoherence(self, lc):

        bispec = np.zeros([self.nfft, self.nfft], dtype=complex)
        mask = hankel(np.arange(self.nfft),
                      np.array([self.nfft - 1] + list(range(self.nfft - 1))))
        # hankel serves  the function of summed frequncy in bispectra
        pseg = np.arange(self.m).transpose()
        sklsq = np.zeros([self.nfft, self.nfft], dtype=complex)
        # k and l frequncies and their square summed over all segments
        skplsq = np.zeros([self.nfft, self.nfft], dtype=complex)
        # frequency at k plus l and its square summed over all segments

        for i in range(self.k):
            iseg = lc.counts[pseg].reshape(1, -1)
            Xf = scipy.fftpack.fft(iseg - np.mean(iseg), self.nfft) / self.m
            CXf = np.conjugate(Xf).ravel(order='C')
            sklsq = sklsq + (Xf * np.transpose(Xf))**2
            skplsq = skplsq + (CXf[mask].reshape(self.nfft, self.nfft)**2)
            bispec = bispec + (Xf * np.transpose(Xf)) * \
                CXf[mask].reshape(self.nfft, self.nfft)
            pseg = pseg + self.m

        bispec = bispec / self.k
        sklsq = sklsq / self.k
        skplsq = skplsq / self.k
        bicoh = (bispec**2) / (sklsq * skplsq)

        return bicoh