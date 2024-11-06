import h5py
import numpy as np
import sys
from DAS_filtering import moving_average,define_butterworth_highpass,filtering,spectrogram
from numpy.fft import fft, ifft


def load_single_DAS_file(file_name):
    hf = h5py.File(file_name, 'r')
    n1 = hf.get('DAS')
    n2 = np.array(n1)
    n2 = n2 * (np.pi / 2 ** 15)
    print(f'[HDF5 Processing] Integrate')
    n22 = np.cumsum(n2,axis=0)
    return n22

def list_hdf5_files_in_dir (file_path):
    #file_path = pathlib.Path("C:\Kate captures/2024-04-18/100 lpm 60s 2")
    list(file_path.iterdir())
    file_count = 0
    for item in file_path.iterdir():
        if item.is_file():
            if file_count == 0:
                file_names = [item]
            else:
                file_names.append(item)
            file_count = file_count + 1
    print('[HDF5 Processing] Number of Files', file_count)
    return file_names

def load_multi_DAS_file(file_names, channels):
    count = 0
    data_DAS = np.zeros((len(file_names) * 200000, channels))

    for i in range(len(file_names)):
        file_name = file_names[i]
        initial_filter_conditions = None
        downsampling_factor = 1
        timeFF = float(file_names[i].name[25:30])
        sys.stdout.flush()
        print('[HDF5 Processing] Current Filename', file_name, flush=True)
        hf = h5py.File(file_name, 'r')
        n1 = hf.get('DAS')
        n2 = np.array(n1)
        n2 = n2 * (np.pi / 2 ** 15)
        print(f'[HDF5 Processing] Integrate')
        n22 = np.cumsum(n2, axis=0)
        Fs = 20000
        filter_cutoff_frequency = 5
        phase_filtered_total = np.array([])
        initial_filter_conditions = None

        phase_filtered_total = filtering(n22, phase_filtered_total, initial_filter_conditions, Fs,
                                         filter_cutoff_frequency)

        data_DAS[count:count + 200000, :] = phase_filtered_total
        count += 200000
    return data_DAS


def generate_training_set(leak_time_period, leak_channels_period,channels_search_period, overlap_rate,DAS_fitlered_data):
    number_of_training_samples = int((DAS_fitlered_data.shape[0]/overlap_rate)*(channels_search_period[1]-channels_search_period[0]))
    training_data = np.zeros((number_of_training_samples, 1, 2901))
    training_label = np.zeros(number_of_training_samples)
    count = 0

    for i in range(channels_search_period[0], channels_search_period[1], 1):

        for j in range(0, DAS_fitlered_data.shape[0], overlap_rate):

            sample = DAS_fitlered_data[j:j + 200000, i]
            DAS_10_second_in_Frequency = fft(sample)
            DAS_10_second_in_Frequency = np.abs(DAS_10_second_in_Frequency[40:10002])
            DAS_10_second_in_Frequency = DAS_10_second_in_Frequency / DAS_10_second_in_Frequency.sum()
            sd = np.expand_dims(DAS_10_second_in_Frequency, axis=0)
            sss = np.concatenate((sd, sd), axis=0)
            ssss = np.expand_dims(sss, axis=1)
            MAtestsss = moving_average(ssss[0, 0, 0:3000], n=100)
            MAtestsss = np.expand_dims(MAtestsss, axis=0)
            MAtestsss = MAtestsss / MAtestsss.sum()
            MAtestsss = np.concatenate((MAtestsss, MAtestsss), axis=0)
            MAtestsss = np.expand_dims(MAtestsss, axis=1)
            # pred = model.predict(MAtestsss)
            training_data[count, 0, :] = MAtestsss[0, 0, :]

            if i > leak_channels_period[0] and i < leak_channels_period[1] and j > leak_time_period[0] and j < leak_time_period[1]:
                training_label[count] = 1
                count += 1
            else:
                training_label[count] = 0
                count += 1

    return training_data,training_label



def generate_training_set_spectrogram(DAS_fitlered_data, channel_start, channel_end,capture_period,fs):
    #channel_start = 140
    #channel_end = 285
    #capture_period = [300000, 2182000]

    spectro_data = []
    for time in range(capture_period[0], capture_period[1] - 20000,20000):
        for channel in range(channel_start, channel_end, 5):
            spectro_5_channels = np.zeros((5, 29, 89))
            count = 0
            for channel_for_spectro in range(5):
                x = DAS_fitlered_data[time:time + 20000, channel + channel_for_spectro]
                spectrogram_sample = spectrogram(x, fs)
                a = spectrogram_sample[0:29, :]
                row_sums = a.sum(axis=1)
                new_matrix = a / row_sums.sum()
                spectro_5_channels[count, :, :] = new_matrix
                count += 1

            spectro_data.append(spectro_5_channels)
        #if time % 100000 == 0:
            print(time)

    spectro_data_numpy = np.array(spectro_data)

    return spectro_data_numpy

