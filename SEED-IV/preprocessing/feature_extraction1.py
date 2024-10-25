import numpy as np
import mne
import scipy.io
from scipy.signal import welch
# preprocessing SEED_IV raw data

mat = scipy.io.loadmat('preprocessing/1_20160518.mat') # load raw data
trial_names = list(mat.keys()) # list all the variables/trials ('cz_eeg')

# get rid of MATLAB metadata
trial_names = [name for name in trial_names if not name.startswith('__')]

original_sampling_rate = 1000
target_sampling_rate = 200

window_length = 4
n_fft = target_sampling_rate * window_length # Fourier transform points based on sliding window
noverlap = 0

psd_trials = {}

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

    # Segment the data into windows of 4 seconds (no overlap)
    segment_length = target_sampling_rate * window_length  # 4 seconds at 200 Hz -> 800 samples
    n_segments = n_times // segment_length  # Number of segments
    eeg_segments = np.array_split(raw.get_data(), n_segments, axis=1)

    # Step 7: Initialize array to store PSD values for all segments
    psd_3d = np.zeros((n_channels, 42, n_segments))  # Placeholder for (channels, frequencies, segments)

    # Step 8: Compute PSD for each segment
    for seg_idx, segment in enumerate(eeg_segments):
        psd_segment = []
        for ch_idx in range(n_channels):
            # Compute the PSD using Welch's method
            freqs, psd = welch(
                segment[ch_idx, :],  # Data for the current channel and segment
                fs=target_sampling_rate,
                nperseg=n_fft,  # Number of points per segment (4s window)
                noverlap=noverlap
            )
            psd_segment.append(psd[:42])  # Keep only the first 42 frequency bins (if needed)
        
        psd_3d[:, :, seg_idx] = np.array(psd_segment)  # Store PSD for this segment

    psd_trials[f'psd{i}'] = psd_3d
    print(f"Processed {trial_name}: PSD 3D shape = {psd_3d.shape}")

scipy.io.savemat('preprocessing/psd_features.mat', psd_trials)
print("PSD features saved to 'psd_features.mat'")