import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

# Main preprocessing function that implements all steps and gives some feedback.
def preprocess_meg(data, fs=2034, z_thresh=5.0):

    data = detrend(data, axis=1, type='linear')

    data = bandpass_filter(data, fs=fs)

    data = remove_channel_outliers(data, z_thresh=z_thresh)

    # Optional final z-score (after filtering & outlier cleanup)
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_std = np.std(data, axis=1, keepdims=True)
    data = (data - data_mean) / (data_std + 1e-8)

    return data

def bandpass_filter(data, lowcut=1, highcut=40, fs=2034, order=4):
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')

    # TEMP normalization to prevent instability, this is safer than doing Z-score before bandpass
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_std = np.std(data, axis=1, keepdims=True) + 1e-8
    data_norm = (data - data_mean) / data_std

    try:
        filtered = filtfilt(b, a, data_norm, axis=1)
    except Exception as e:
        print("  ⚠️ Filtering failed:", e)
        raise

    # Optionally denormalize (not needed if you z-score later, that is why this is safer)
    return filtered

def remove_channel_outliers(data, z_thresh=5.0):
    cleaned = np.copy(data)

    for ch in range(data.shape[0]):
        ch_mean = np.mean(data[ch])
        ch_std = np.std(data[ch])
        z_scores = (data[ch] - ch_mean) / ch_std
        outliers = np.abs(z_scores) > z_thresh

        num_outliers = np.sum(outliers)
        if num_outliers > 0:
            print("Outliers detected:", num_outliers, "for channel", ch)
        
        if np.any(outliers):
            valid = ~outliers
            cleaned[ch, outliers] = np.interp(
                np.flatnonzero(outliers),
                np.flatnonzero(valid),
                data[ch, valid]
            )
    return cleaned

def plot_meg_signals(raw_data, clean_data, fs=2034, channels=[0, 1, 2], duration=2):
    time = np.arange(raw_data.shape[1]) / fs
    samples = int(duration * fs)

    plt.figure(figsize=(12, len(channels) * 3))

    print("Important to note that the following plots are not comparable in scale. Only the shape")

    for i, ch in enumerate(channels):
        plt.subplot(len(channels), 1, i + 1)

        # Rescale for visibility (because the raw data is so small)
        raw = raw_data[ch, :samples] * 1e12
        clean = clean_data[ch, :samples] * 1 # Keep the same to make it visible

        plt.plot(time[:samples], raw, label='Raw', alpha=0.5)
        plt.plot(time[:samples], clean, label='Cleaned', alpha=0.8)
        plt.title(f"Channel {ch}")
        plt.ylabel("Signal (pT)")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()