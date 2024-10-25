import numpy as np
import mne
from scipy.io import loadmat, savemat

# Load the downsampled and filtered EEG data
data = loadmat('SEED-IV/preprocessing/downsampled_and_filtered.mat')

# Assuming the structure is similar to before: 24 trials, each of shape (62 channels, n_samples)
trials = [data[trial] for trial in data if not trial.startswith('__')]

# Parameters
n_channels = 62
sampling_rate = 200  # Hz
window_size = 4 * sampling_rate  # 4 seconds, 800 samples
overlap = 0  # No overlap as per SEED-IV
n_trials = len(trials)

# Frequency bands definition
freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 31),
    'gamma': (31, 50)
}

# Pre-allocate 3D array: (channels, time windows, frequency bands)
psd_all_trials = []

# Iterate through each trial
for trial in trials:
    n_samples = trial.shape[1]
    n_windows = n_samples // window_size
    psd_trial = np.zeros((n_channels, n_windows, len(freq_bands)))

    # Segment trial into non-overlapping windows
    for i in range(n_windows):
        segment = trial[:, i * window_size:(i + 1) * window_size]

        # Compute PSD for each channel in this window
        for ch in range(n_channels):
            psd, freqs = mne.time_frequency.psd_array_multitaper(segment[ch], sfreq=sampling_rate, fmin=1, fmax=50)

            # Extract power for each frequency band
            for j, (band, (fmin, fmax)) in enumerate(freq_bands.items()):
                band_power = psd[(freqs >= fmin) & (freqs < fmax)].mean()
                psd_trial[ch, i, j] = band_power

    psd_all_trials.append(psd_trial)

# Convert the list of trials into a numpy array
psd_all_trials = np.array(psd_all_trials)

# Save the results to a .mat file
savemat('psd_results.mat', {'psd_data': psd_all_trials})
