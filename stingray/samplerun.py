import numpy as np
from stingray import Lightcurve
from stingray import Bispectrum


def run():

    tstart = 0.0
    tend = 0.64
    dt = 0.0001
    time = np.linspace(tstart, tend, int((tend - tstart) / dt))
    # total number of samples = 6400
    # noise = np.random.normal(0, 1, 6400)
    count = np.sin((2 * np.pi * time) / 0.64)
    lc = Lightcurve(time, counts=count)
    bs = Bispectrum(lc=lc, m=64, nfft=64)
    bs.lc = lc
    bs.plot()

if __name__ == '__main__':
    run()
