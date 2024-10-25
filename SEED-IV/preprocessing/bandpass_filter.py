import numpy as np
import mne
import scipy.io
# preprocessing SEED_IV raw data

# load raw data
mat = scipy.io.loadmat('SEED-IV/preprocessing/1_20160518.mat')

# list all the variables/trials ('cz_eeg')
trial_names = list(mat.keys())

# get rid of MATLAB metadata
trial_names = [name for name in trial_names if not name.startswith('__')]

original_sampling_rate = 1000
target_sampling_rate = 200

processed_trials = {}

for i, trial_name in enumerate(trial_names, start=1):
    eeg_data = mat[trial_name]  # access data for single trial, channels x time

    # needed for MNE object
    n_channels, n_times = eeg_data.shape
    channel_names = [f'EEG {j+1}' for j in range(n_channels)]  # list of channel names
    channel_types = ['eeg'] * n_channels  # set channel types to 'eeg'

    info = mne.create_info(ch_names=channel_names, sfreq=original_sampling_rate, ch_types=channel_types)
    raw = mne.io.RawArray(eeg_data, info)

    raw.resample(sfreq=target_sampling_rate) # downsample to 200 Hz
    raw.filter(l_freq=1, h_freq=75) # bandpass filter btw 1-75 Hz

    filtered_data = raw.get_data() # load filtered and downsampled data 
    processed_trials[f'trial{i}'] = filtered_data # store filtered and downsampled data

    print(f"Processed {trial_name}: {filtered_data.shape}")

scipy.io.savemat('SEED-IV/preprocessing/downsampled_and_filtered.mat', processed_trials)
print("Data saved to 'downsampled_and_filtered.mat'")