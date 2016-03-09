#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.fftpack
import stingray.lightcurve as lightcurve
import stingray.utils as utils
from stingray.utils import simon
__all__ = ["bispectra"]


class bispectra(object):

  def __init__(self, lc=None,m=None, nfft=None):
    """
   
   
    Parameters 
    --------
    lc: lightcurve.Lightcurve object, optional, default None
            The light curve data to be Fourier-transformed.
    
    m : Number of samples in each segment Default :64(expecting huge data)

    nfft : Fft size Default is 128 
 

    attributes :
    ----------
    k: Number of segments

    m : Number of samples in each segment Default :64(expecting huge data)

    nfft : Fft size Default is 128

    nsamp  : total numberof samples from the input signal 


    """

    if lc is not None :
      pass
    else :
        self.k=None
        self.nfft=None
        self.m=None
        self.nsamp=None
        return 

    lc.counts = lc.counts.ravel(order='F')
    self.nsamp  =lc.counts.shape[0]

    if m is None :
      self.m =64
    else :
      assert isinstance(m, int ), "m is not a int !"

    if nfft is None :
      self.nfft =128
    else :
      assert isinstance(nfft, int), "nfft is not a string!"
    
    self.k= self.nsamp//self.m
    return self.compute_bispectra(lc)

  def compute_bispectra(self,lc):

  
    

    Bispec = np.zeros([self.nfft,self.nfft]) 
    mask = hankel(np.arange(self.nfft),np.array([self.nfft-1]+range(self.nfft-1)))
    #here hankel serves  the function of summed frequncy which is used in bispectra 
    pseg = np.arange(self.m).transpose()
    
    
   
    
  
    for i in xrange(self.k):
      iseg = lc.counts[pseg].reshape(1,-1)
      Xf = scipy.fftpack.fft(iseg - np.mean(iseg), self.nfft) / self.m
      CXf = np.conjugate(Xf).ravel(order='F')
   
      temp=Bispec.reshape(1,-1)
      temp=(Xf * np.transpose(Xf)) * CXf[mask].reshape(self.nfft, self.nfft)
      Bispec = Bispec + \
      temp.reshape(Bispec.shape)
      pseg = pseg + int(self.m)

    Bispec = scipy.fftpack.fftshift(Bispec) / self.k


 
    return Bispec


