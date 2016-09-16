import numpy as np
import pytest
from stingray import Lightcurve
from stingray import Bispectrum, bicoherence


class TestBicoherencefunction(object):

    def setup_class(self):
        tstart = 0.0
        tend = 0.64
        dt = 0.0001
        time = np.linspace(tstart, tend, int((tend - tstart) / dt))
        # total number of samples = 6400
        # noise = np.random.normal(0, 1, 6400)
        count = np.sin((2 * np.pi * time) / 0.64)
        self.lc = Lightcurve(time, count)

    def test_bicoherence(self):
        with pytest.raises(TypeError):
            bispec = bicoherence(self.lc.counts)
        # bicoherence doesnt necssarily be less than 1 as in case of coherence


class TestBispecturm(object):

    def setup_class(self):
        tstart = 0.0
        tend = 0.64
        dt = 0.0001
        time = np.linspace(tstart, tend, int((tend - tstart) / dt))
        # total number of samples = 6400
        # noise = np.random.normal(0, 1, 6400)
        count = np.sin((2 * np.pi * time) / 0.64)
        self.lc = Lightcurve(time, count)

    def test_make_empty_bispectra(self):
        bs = Bispectrum()
        assert bs.n == 64
        assert bs.nfft == 128
        assert bs.m is None
        assert bs.ncounts is None

    def test_make_bispectra_from_lightcurve(self):

        bs = Bispectrum(self.lc, n=64, nfft=64)
        assert bs.ncounts == self.lc.counts.shape[0]
        assert bs.bispectrum.shape == (bs.nfft, bs.nfft)

    def test_symmetry1(self):
        """

        References
        ----------
        .. [1] Bispectrum estimation: A digital signal processing framework
                    C.L. Nikias ; M.R. Raghuveer   DOI: 10.1109/PROC.1987.13824 

            More explicit properties on Pgno:873 of reference [1]


        Exploting the fact that bispectrum is symmetric nature of  B(w1,w2) == B(w2,w1)

        """

        bs = Bispectrum(self.lc, n=64, nfft=64)
        for i in range(bs.nfft):
            for j in range(bs.nfft):
                assert abs(bs.bispectrum[i, j]) == abs(bs.bispectrum[j, i])

    def test_symmetry2(self):
        """
        References
        ----------
        .. [1] Bispectrum estimation: A digital signal processing framework
                    C.L. Nikias ; M.R. Raghuveer   DOI: 10.1109/PROC.1987.13824 

            More explicit properties on Pgno:873 of reference [1]


         this [B(w1,w2) == B(w1,-w1-w2) ] along with symmetry property 1 gives bispectrum 12 sector symmetry 

        """
        bs = Bispectrum(self.lc, n=64, nfft=64)
        for i in range(bs.nfft):
            for j in range(bs.nfft):
                if i != 0 and j != 0:
                    x = (bs.nfft - j - i)
                    assert abs(bs.bispectrum[i, j]) == abs(
                        bs.bispectrum[i + x, j + x])

    def test_init_with_lightcurve(self):
        assert Bispectrum(self.lc)

    def test_init_without_lightcurve(self):
        with pytest.raises(TypeError):
            bispec = Bispectrum(self.lc.counts)

    def test_init_with_nonsense_data(self):
        with pytest.raises(TypeError):
            nonsense_data = [None for i in range(100)]
            bispec = Bispectrum(nonsense_data)

    def test_init_with_nfft_not_int(self):
        with pytest.raises(TypeError):
            bispec = Bispectrum(self.lc, nfft="bla")

    def test_Bichorence(self):
        bs = Bispectrum(self.lc, n=64, nfft=64)
        bicoherence = bs.bicoherence()
        assert bicoherence.shape == (bs.nfft, bs.nfft)
