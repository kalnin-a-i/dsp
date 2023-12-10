from scipy.fft import fft, ifft, fftfreq
import numpy as np


def get_equalizer(sweeper_original, sweeper_speaker, num_bands=32, ):

    # max freq defined from Nyquist theorem
    max_freq = sweeper_original[1] // 2

    # get fourier of both signals
    freqs_original, fft_original = fftfreq(len(sweeper_original[0]), 1 / sweeper_original[1]), np.abs(fft(sweeper_original[0]))
    freqs_speaker, fft_speaker = fftfreq(len(sweeper_speaker[0]), 1 / sweeper_speaker[1]), np.abs(fft(sweeper_speaker[0]))

    # create frequency bands
    bands = np.linspace(1, max_freq, num_bands)

    # cut freq by max freq
    freqs_original, freqs_speaker = freqs_original[freqs_original < max_freq], freqs_speaker[freqs_speaker < max_freq] 

    # get mean fft vulues inside each band
    
    mean_original = np.zeros(len(bands))
    mean_speaker = np.zeros(len(bands))
    for i in range(1, len(bands)):
        mean_original[i] = fft_original[(freqs_original < bands[i]) & ((freqs_original >  bands[i-1]))].mean()
        mean_speaker[i] = fft_speaker[(freqs_speaker < bands[i]) & ((freqs_speaker > bands[i-1]))].mean()
    
    eqaulizer_coefs = mean_speaker / mean_original
    
    return eqaulizer_coefs, bands


def apply_equaizer(speaker_signal, equalizer_coefs, bands):

    freqs_speaker, fft_speaker = fftfreq(len(speaker_signal[0]), 1 / speaker_signal[1]),\
                                fft(speaker_signal[0])
    
    for i in range(1, len(bands)):
        fft_speaker[(freqs_speaker < bands[i]) & ((freqs_speaker > bands[i-1]))] /= equalizer_coefs[i]
        
    equalized_signal = ifft(fft_speaker) 
    return equalized_signal.real, fft_speaker