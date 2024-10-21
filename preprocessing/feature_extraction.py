import numpy as np
import mne
from scipy.signal import welch as signal
import scipy.io

# Load the downsampled data from the .mat file
data = scipy.io.loadmat('preprocessing/downsampled_and_filtered.mat')

trial_names = [f'trial{i+1}' for i in range(24)]  # Modify based on actual names in the file

# Store the downsampled trials in a list
downsampled_data = []

for trial_name in trial_names:
    # Extract each trial data from the loaded .mat file
    trial_data = data[trial_name]
    downsampled_data.append(trial_data)

# Convert to a NumPy array if needed (optional)
downsampled_data = np.array(downsampled_data)  # Shape: (24, 62, n_samples)


# Frequency bands
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 31),
    'gamma': (31, 50)
}

# Sampling rate after downsampling
sfreq = 200  # Hz

# Window size for each segment (4 seconds)
window_size = 4 * sfreq  # 800 samples for 4 seconds

# Initialize an empty list to store PSD data for all trials
psd_all_trials = []

for trial_data in downsampled_data:  # Assuming 'downsampled_data' is a list of downsampled EEG trials
    n_channels, n_samples = trial_data.shape

    # Split the trial data into 4-second windows without overlap
    n_windows = n_samples // window_size  # Number of full windows in the data
    trial_psd = np.zeros((n_channels, n_windows, len(bands)))  # Initialize a 3D array for this trial
    
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        window_data = trial_data[:, start:end]
        
        # Compute the PSD for each channel using Welch's method
        for ch in range(n_channels):
            freqs, psd = signal.welch(window_data[ch, :], fs=sfreq, nperseg=window_size)
            
            # Extract PSD for each frequency band
            for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
                band_power = np.trapz(psd[(freqs >= fmin) & (freqs <= fmax)], freqs[(freqs >= fmin) & (freqs <= fmax)])
                trial_psd[ch, w, i] = band_power
    
    psd_all_trials.append(trial_psd)  # Append PSD of this trial to the list

# Convert the list to a numpy array
psd_all_trials = np.array(psd_all_trials)  # Shape: (n_trials, n_channels, n_windows, n_bands)

scipy.io.savemat.savemat('psd_data.mat', {'psd_all_trials': psd_all_trials})