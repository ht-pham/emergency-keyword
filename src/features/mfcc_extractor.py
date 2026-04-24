import os
import numpy as np
import scipy
import scipy.io.wavfile as wav
import scipy.fftpack as fft
from scipy.signal import get_window
from scipy.fftpack import dct

wav_data_dir = './../../data/wav/'

class MFCCExtractor:
    def __init__(self, sample_rate=16000, frame_size=0.025, frame_stride=0.01, num_filters=40, num_ceps=12):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.num_filters = num_filters
        self.num_ceps = num_ceps

    def emphasize_and_frame(self, file_path):
        # Read the audio file
        self.sample_rate, signal = wav.read(file_path)
        
        # Pre-emphasis
        emphasized_signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
        
        # Framing
        frame_len = int(self.frame_size * self.sample_rate)
        frame_step = int(self.frame_stride * self.sample_rate)
        signal_len = len(signal)
        num_frames = int(np.ceil(float(np.abs(signal_len - frame_len)) / frame_step))
        
        pad_signal_length = num_frames * frame_step + frame_len
        z = np.zeros((pad_signal_length - signal_len))
        pad_signal = np.append(emphasized_signal, z)

        # Slice signal into frames using indices
        ## make a single frame [0,1,2,...,frame_len-1]
        single_frame = np.arange(0, frame_len)
        ## duplicate the single frame num_frames times
        ### output: [[0,1,2,...,frame_len-1],
        ###          [0,1,2,...,frame_len-1],
        ###          ...] (num_frames rows)
        n_frames = np.tile(single_frame, (num_frames, 1))

        ## create starting indices for each frame
        frame_idx= np.arange(0, num_frames * frame_step, frame_step)
        ### repeat frame_idx for frame_len times and transpose to get the same shape as n_frames
        frame_idx = np.tile(frame_idx, (frame_len, 1)).T

        row_indices = n_frames + frame_idx
        frames = pad_signal[row_indices.astype(np.int32, copy=False)]

        frames *= np.hamming(frame_len) 

        return frames

    def ttf_conversion(self,frames,NFFT=512):
        # Compute the magnitude spectrum of each frame using FFT
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        # Compute the power spectrum of each frame
        pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
        return pow_frames
    
    def mel_filter_bank(self, pow_frames,NFFT=512):
        # mel scale = 2595*log10(1+frequency/700)
        mel_scale = lambda f: 2595 * np.log10(1+f/700)
        # Convert the frequency range to mel scale
        hz_points = lambda m: 700*(10**(m/2595)-1)

        mels = np.linspace(mel_scale(0), mel_scale(self.sample_rate/2), self.num_filters + 2)
        freqs = hz_points(mels)

        # Scale the frequencies to corresponding bin indices in the FFT output
        bins = np.floor((NFFT + 1) * freqs / self.sample_rate).astype(int)

        # Create the filters for each frequency bin
        filters = np.zeros((self.num_filters,int(np.floor(NFFT/2 +1))))
        for i in range(1, self.num_filters + 1):
            left = int(bins[i-1])
            center = int(bins[i])
            right = int(bins[i+1])
            for j in range(left, center):
                filters[i-1, j] = (j - bins[i-1]) / (bins[i] - bins[i-1])
            for j in range(center, right):
                filters[i-1, j] = (bins[i+1] - j) / (bins[i+1] - bins[i])

        # Create the filter bank by computing the dot product of the power spectrum and the filters
        filter_banks = np.dot(pow_frames, filters.T)
        # Avoid log of zero
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks) 
        
        return filter_banks
    
    def logarithmic_scale(self, filter_banks):
        # Take the logarithm of the filter bank energies
        log_filter_banks = 20*np.log10(filter_banks)
        return log_filter_banks
    
    def dct(self,log_filter_banks):
        # Compute the MFCCs by applying DCT to the log filter bank energies
        mfccs = dct(log_filter_banks, type=2, axis=1, norm='ortho')[:,:self.num_ceps]
        return mfccs
    
