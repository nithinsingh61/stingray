import numpy as np

from nose.tools import raises

from stingray import Lightcurve
from stingray import Bispectrum
import matplotlib.pyplot as plt


class TestBispectra(object):

    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 0.64
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend - tstart) / dt))
        # total number of samples = 6400
        # noise = np.random.normal(0, 1, 6400)
        cls.count = np.sin((2 * np.pi * time) / 0.64)
        cls.lc = Lightcurve(time, counts=cls.count)

    def test_make_empty_bispectra(self):
        bs = Bispectrum()
        assert bs.m == 64
        assert bs.nfft == 128
        assert bs.k is None
        assert bs.nsamp is None

    def test_make_bispectra_from_lightcurve(self):

        bs = Bispectrum(self.lc, m=64, nfft=64)
        bispectrum = bs.compute_bispectrum(self.lc)
        print bispectrum
        assert bs.nsamp == self.lc.counts.shape[0]
        if bs.nfft % 2 == 0:
            freaxis = np.transpose(
                np.arange(-1 * bs.nfft / 2, bs.nfft / 2))
        else:
            freaxis = np.transpose(
                np.arange(-1 * (bs.nfft - 1) / 2, (bs.nfft - 1) / 2 + 1))
            # cont1 = plt.contour(abs(Bspec), 4, waxis, waxis)
        cont = plt.contourf(
            freaxis, freaxis, abs(bispectrum), 200, cmap=plt.cm.Spectral_r)
        plt.colorbar(cont)
        plt.title('Bispectrum ')
        plt.xlabel('f1')
        plt.ylabel('f2')
        # plt.show()

    def test_symmetry1(self):
        # B(w1,w2) == B(w2,w1)
        bs = Bispectrum(self.lc, m=64, nfft=64)
        bispectrum = bs.compute_bispectrum(self.lc)
        for i in range(bs.nfft):
            for j in range(bs.nfft):
                assert abs(bispectrum[i, j]) == abs(bispectrum[j, i])

    def test_symmetry2(self):
        # B(w1,w2) == B(w1,-w1-w2)
        bs = Bispectrum(self.lc, m=64, nfft=64)
        bispectrum = bs.compute_bispectrum(self.lc)
        for i in range(bs.nfft):
            for j in range(bs.nfft):
                if i != 0 and j != 0:
                    x = (bs.nfft - j - i)
                    assert abs(bispectrum[i, j]) == abs(
                        bispectrum[i + x, j + x])

    def test_xBichorence(self):

        bs = Bispectrum(self.lc, m=64, nfft=64)
        bicoherence = bs.bicoherence(self.lc)
        print bicoherence
        assert bs.nsamp == self.lc.counts.shape[0]
        if bs.nfft % 2 == 0:
            freaxis = np.transpose(
                np.arange(-1 * bs.nfft / 2, bs.nfft / 2))
        else:
            freaxis = np.transpose(
                np.arange(-1 * (bs.nfft - 1) / 2, (bs.nfft - 1) / 2 + 1))
        cont = plt.contourf(
            freaxis, freaxis, (bicoherence), 200, cmap=plt.cm.Spectral_r)
        plt.colorbar(cont)
        plt.title('Bichorence')
        plt.xlabel('f1')
        plt.ylabel('f2')
        # plt.show()

    def test_init_with_lightcurve(self):
        assert Bispectrum(self.lc)

    @raises(AssertionError)
    def test_init_without_lightcurve(self):
        assert Bispectrum(self.lc.counts)

    @raises(AssertionError)
    def test_init_with_nonsense_data(self):
        nonsense_data = [None for i in range(100)]
        assert Bispectrum(nonsense_data)

    @raises(AssertionError)
    def test_init_with_nonsense_nfft(self):
        nonsense_nfft = "bla"
        assert Bispectrum(self.lc, nfft=nonsense_nfft)
