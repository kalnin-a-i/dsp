from scipy.fft import fft, ifft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

def get_equalizer(sweeper_original,
                  sweeper_original_sample_rate,
                  sweeper_speaker,
                  sweeper_speaker_sample_rate,
                  num_bands=32,
                  plot=False):

    # max freq defined from Nyquist theorem
    max_freq = sweeper_original_sample_rate // 2

    # get fourier of both signals
    freqs_original, fft_original = fftfreq(len(sweeper_original), 1 / sweeper_original_sample_rate), np.abs(fft(sweeper_original, norm='ortho'))
    freqs_speaker, fft_speaker = fftfreq(len(sweeper_speaker), 1 / sweeper_speaker_sample_rate), np.abs(fft(sweeper_speaker, norm='ortho'))

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

    if plot:
        n, m = fft_original.shape[0], fft_speaker.shape[0]
        plt.plot(freqs_original[:n //2], fft_original[:n //2])
        plt.plot(freqs_speaker[:m //2], fft_speaker[:m //2])
        plt.xlabel('frequancy, Hz')
        plt.ylabel('amplitude')

    return eqaulizer_coefs, bands


def apply_equalizer(signal, signal_sample_rate, equalizer_coefs, bands, plot=False):

    freqs, signal_fft = fftfreq(len(signal), 1 / signal_sample_rate),\
                                fft(signal, norm='ortho')
    
    if plot:
        n = signal_fft.shape[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[1].plot(freqs[: n // 2], signal_fft[: n // 2], label='original fft')
    
    for i in range(1, len(bands)):
        signal_fft[(np.abs(freqs) < bands[i]) & (np.abs(freqs) > bands[i-1])] /= equalizer_coefs[i]

    equalized_signal = ifft(signal_fft, norm='ortho').real

    if plot:
    
        axes[0].plot(equalized_signal, label='equalized siganal')
        axes[0].plot(signal, label='original signal')
        axes[0].legend()
        axes[0].set_xlabel('time')
        axes[1].set_ylabel('amplitude')

        axes[1].plot(freqs[: n // 2], signal_fft[: n // 2], label='equalized fft')
        axes[1].legend()
        axes[1].set_xlabel('frequency, Hz')
        axes[1].set_ylabel('amplitude')


    return equalized_signal, signal_fft