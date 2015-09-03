#!/usr/bin/env python

import filterbank
from filterbank import FilterbankFile
from psrfits import PsrfitsFile
from psrfits2fil import translate_header
import psr_utils
import numpy as np


pack_4bit = lambda xs: xs[::2] + (xs[1::2] << 4)
unpack_4bit = lambda xs: np.dstack([xs & 15, xs >> 4]).flatten()

class GeneralFile:

    def __init__(self, fn, *args):
        if '.fil' in fn:
            if 'update' in args:
                args.remove('update')
                self.base = FilterbankFile(fn, 'readwrite', *args)
            elif 'readonly' in args:
                args.remove('readonly')
                self.base = FilterbankFile(fn, 'read', *args)
            else:
                self.base = FilterbankFile(fn, *args)
        elif '.fits' in fn:
            if 'readwrite' in args:
                args.remove('readwrite')
                self.base = PsrfitsFile(fn, 'update', *args)
            elif 'read' in args:
                args.remove('read')
                self.base = PsrfitsFile(fn, 'readonly', *args)
            else:
                self.base = PsrfitsFile(fn, *args)
            self.header = translate_header(self.base)
            self.header['nbits'] = self.base.nbits
            self.dt = self.tsamp
            self.frequencies = \
                    self.freqs if self.freqs[0] > self.freqs[1] else self.freqs[::-1] 
            self.nspec = self.nsubints*self.nsamp_per_subint

    def __getattr__(self, name):
        try:
            if isinstance(self.base,FilterbankFile):
                val = self.base.__getattr__(name)
            elif name in self.header:
                val = self.header[name]
            else:
                raise ValueError("No attribute called '%s'" % name)
        except:
            try:
                val = self.__getattribute__(name)
            except:
                val = self.base.__getattribute__(name)
        return val

    
    def get_timeslice(self, start, stop):
        """
        Get spectra given start and stop time in seconds.
        """
        start_bins = int(np.round(start/self.dt))
        stop_bins = int(np.round(stop/self.dt))
        return self.get_spectra(start_bins,stop_bins)
    
    def get_spectra(self,start,stop,applyChT=True):
        """
        Get spectra given start and stop indices.
        If psrfits, decide whether channel transforms are applied. 
        """
        if isinstance(self.base,FilterbankFile):
            return self.base.get_spectra(start,stop)
        elif isinstance(self.base,PsrfitsFile):
            first_sub = int(start)/self.nsamp_per_subint
            second_sub = int(stop-1)/self.nsamp_per_subint
            start_idx = int(start)%self.nsamp_per_subint
            stop_idx = int(stop)%self.nsamp_per_subint
            stop_idx = None if stop_idx == 0 else stop_idx
              
            if first_sub >= self.nsubints or first_sub == 1:
                return np.zeros((0,0))
            if second_sub == self.nsubints:
                second_sub -= 1
                stop_idx = None
            if first_sub == second_sub:
                data = self.base.read_subint(first_sub,applyChT,applyChT,applyChT)
                return data[start_idx:stop_idx]
            else:
                data1 = self.base.read_subint(first_sub,applyChT,applyChT,applyChT)
                data2 = self.base.read_subint(second_sub,applyChT,applyChT,applyChT)
                return np.vstack([data1[start_idx:],data2[:stop_idx]])

    def write_spectra(self, spectra, ispec,applyChT=True):
        if isinstance(self.base, FilterbankFile):
            self.base.write_spectra(spectra, ispec)
        elif isinstance(self.base, PsrfitsFile):
            nspec, nchans = spectra.shape
            if nchans != self.nchans:
                raise ValueError("Cannot write spectra. Incorrect shape. " \
                        "Number of channels in file: %d Number of " \
                        "channels in spectra to write: %d" % \
                        (self.nchans, nchans))
            if ispec >= self.nspec:
                raise NotImplementedError("Cannot write beyond end of file atm.")
            first_sub = int(ispec)/self.nsamp_per_subint
            second_sub = int(ispec+nspec-1)/self.nsamp_per_subint
            start_idx = int(ispec)%self.nsamp_per_subint
            stop_idx = int(ispec+nspec)%self.nsamp_per_subint
            stop_idx = None if stop_idx == 0 else stop_idx
            if second_sub == self.nsubints:
                second_sub = first_sub
                stop_idx = None
            if first_sub == second_sub:
                data = self.base.read_subint(first_sub,False,False,False)
                if applyChT: spectra = remove_ch_transforms(self,spectra,first_sub)
                data[start_idx:stop_idx] = spectra
                data = data.flatten()
                if self.nbits == 4:
                    data = pack_4bit(data)
                self.fits['SUBINT'].data[first_sub]['DATA'] = data
            else:
                data1 = self.base.read_subint(first_sub,False,False,False)
                data2 = self.base.read_subint(second_sub,False,False,False)
                if applyChT:
                    spectra1 = remove_ch_transforms(self,spectra[:-stop_idx],first_sub)
                    spectra2 = remove_ch_transforms(self,spectra[-stop_idx:],second_sub)
                else:
                    spectra1 = spectra[:-stop_idx]
                    spectra2 = spectra[-stop_idx:]
                data1[start_idx:] = spectra1
                data2[:stop_idx] = spectra2
                data1 = data1.flatten()
                data2 = data2.flatten()
                if self.nbits == 4:
                    data1 = pack_4bit(data1)
                    data2 = pack_4bit(data2)
                self.fits['SUBINT'].data[first_sub]['DATA'] = data1
                self.fits['SUBINT'].data[second_sub]['DATA'] = data2
    
    def append_spectra(self,spectra):
        """
        Append spectra to the file if it is filterbank and not read-only.
        """ 
        if isinstance(self.base, FilterbankFile):
            self.base.append_spectra(spectra)
        else:
            raise NotImplementedError("Appending only supported for .fil files")

    def make_clone(self, outfn):
        """
        Make a copy of the possibly updated source file.
        """
        if isinstance(self.base, FilterbankFile):
            create_filterbank_file(outfn, self.header, \
                     self.get_spectra(0,self.nspec),self.nbits)
        elif isinstance(self.base, PsrfitsFile):
            self.fits.writeto(outfn)
    
    def dedisperse(self, DM):
        """
        Dedisperse entire spectra and return timeseries.
        """
        spectra = self.get_spectra(0,self.nspec)
        fs = self.frequencies
        delays = psr_utils.delay_from_DM(DM,fs)
        delays -= delays.min()
        delays = np.round(delays/self.dt)
        B = np.zeros(self.nspec,spectra.dtype)
        for i,d in enumerate(delays):
            B += np.hstack((spectra[d:,i],spectra[:d,i]))
        return B

def create_filterbank_file(outfn, header, spectra=None, nbits=8, \
                           verbose=False, mode='append'):
    if '.fil' in outfn:
        fil = filterbank.create_filterbank_file(outfn, header, spectra=spectra, \
                nbits=nbits, verbose=verbose, mode=mode)
        fil.close()
        return GeneralFile(outfn,mode)
    else:
        raise ValueError("Out-filename must have '.fil' extension")


def remove_ch_transforms(fitsf,spectra,isub):
    """
    Reverses the channel weightings, scales, and offsets (given in psrfits file).
    """
    if isinstance(fitsf, PsrfitsFile) or isinstance(fitsf.base, PsrfitsFile):
        w = fitsf.get_weights(isub)
        s = fitsf.get_scales(isub)
        o = fitsf.get_offsets(isub)
        w[ w <= 0. ] = 1.
        s[ s <= 0. ] = 1.
        return np.clip(((spectra / w - o) / s), 0, 255).astype('uint8')
    else:
        raise NotImplementedError("Only implemented for psrfits files")

