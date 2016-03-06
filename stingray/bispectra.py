#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.fftpack
#import stingray.lightcurve as lightcurve
#import stingray.utils as utils
#from stingray.utils import simon



def bispectra(lc,m):
 
   #lc is  light curve object 
   # m denotes samples in each segment 

   #assert isinstance(lc, lightcurve.Lightcurve)
    
    #lc must be a lightcurve.Lightcurve object!"
  
  lc.counts = lc.counts.ravel(order='F')
  nfft=128  # For now I have chosen  128

  Bispec = np.zeros([nfft,nfft]) 
  mask = hankel(np.arange(nfft),np.array([nfft-1]+range(nfft-1)))
  #here hankel serves  the function of summed frequncy which is used in bispectra 
  pseg = np.arange(m).transpose()
    
  k=lc.counts.shape[0]//m
   
  # k segments m samples each 
  
  for i in xrange(k):
    iseg = lc.counts[pseg].reshape(1,-1)
    Xf = scipy.fftpack.fft(iseg - np.mean(iseg), nfft) / m
    CXf = np.conjugate(Xf).ravel(order='F')
   
    temp=Bispec.reshape(1,-1)
    temp=(Xf * np.transpose(Xf)) * CXf[mask].reshape(nfft, nfft)
    Bispec = Bispec + \
    temp.reshape(Bispec.shape)
    pseg = pseg + int(m)

  Bispec = scipy.fftpack.fftshift(Bispec) / k


 
  return Bispec


