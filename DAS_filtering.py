import numpy as np
from scipy.signal import butter, lfilter
from scipy import signal

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# filtering
def define_butterworth_highpass(cutoff, Fs):
    nyquist = Fs / 2
    order = 3
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', output='ba', analog=False)
    sos = butter(order, normal_cutoff, btype='high', output='sos', analog=False)
    return b, a, sos

def filtering(phase, phase_filtered_total, initial_filter_conditions, Fs, filter_cutoff_frequency):
    b, a, sos = define_butterworth_highpass(cutoff=filter_cutoff_frequency, Fs=Fs)

    if initial_filter_conditions is None:
        filtered_data = lfilter(b, a, phase, axis=0)
        phase_filtered_total = filtered_data
        initial_filter_conditions = True
    else:
        filtered_data = lfilter(b, a, phase, axis=0)
        phase_filtered_total = np.concatenate((phase_filtered_total, filtered_data), axis=0)

    return phase_filtered_total


def spectrogram(phase_filtered_total, fs):
    f, t, Sxx = signal.spectrogram(phase_filtered_total, fs)
    a = Sxx[0:29, :]
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums.sum()
    return new_matrix



