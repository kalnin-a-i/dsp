import numpy as np
import matplotlib.pyplot as plt

def mixer(signal, noise, snr_db, plot=False):
    target_ratio = 10 ** (-snr_db / 20)


    signal_mean_ampl =  np.abs(signal).mean()
    noise_mean_ampl = np.abs(signal).mean()

    initial_ratio = signal_mean_ampl / noise_mean_ampl

    noise = noise * target_ratio / initial_ratio

    mix = signal + noise

    if plot:
        plt.plot(mix, label='mixed')
        plt.plot(signal, label='signal')
        plt.legend()
        plt.xlabel('time')
    return mix