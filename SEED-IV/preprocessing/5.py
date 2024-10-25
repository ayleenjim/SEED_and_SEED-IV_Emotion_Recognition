import numpy as np
import scipy.signal as signal
from scipy.io import loadmat, savemat

# Load the downsampled and filtered EEG data
data = loadmat('SEED-IV/preprocessing/downsampled_and_filtered.mat')

# Assuming the structure is: 24 trials, each of shape (62 channels, n_samples)
trials = [data[trial] for trial in data if not trial.startswith('__')]

# Parameters
n_channels = 62
sampling_rate = 200  # Hz
window_size = 4 * sampling_rate  # 4 seconds, 800 samples (4s * 200 Hz)
n_trials = len(trials)

# Frequency bands definition
freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 31),
    'gamma': (31, 50)
}

# Get the minimum number of samples across all trials
min_samples = min([trial.shape[1] for trial in trials])

# Pre-allocate storage for the results
psd_all_trials = []

# Welch method parameters
nperseg = window_size  # Size of each segment (4 seconds)
noverlap = 0           # No overlap (as per SEED-IV description)

# Iterate through each trial
for trial in trials:
    # Trim the trial to the same length (min_samples)
    trial_trimmed = trial[:, :min_samples]

    # Initialize PSD array for this trial
    n_segments = min_samples // nperseg
    psd_trial = np.zeros((n_channels, n_segments, len(freq_bands)))

    # Compute PSD for each channel using Welch's method
    for ch in range(n_channels):
        # Welch's method computes PSD for each 4-second window
        freqs, psd = signal.welch(trial_trimmed[ch], fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nperseg)

        # Extract power for each frequency band
        for i, (band, (fmin, fmax)) in enumerate(freq_bands.items()):
            band_power = np.mean(psd[(freqs >= fmin) & (freqs < fmax)], axis=0)
            psd_trial[ch, :, i] = band_power

    # Append the PSD data for this trial
    psd_all_trials.append(psd_trial)

# Convert the list of trials into a numpy array (3D: trials x channels x frequency bands)
psd_all_trials = np.array(psd_all_trials)

# Save the results to a .mat file
savemat('psd_results.mat', {'psd_data': psd_all_trials})
