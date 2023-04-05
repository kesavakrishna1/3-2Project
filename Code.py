import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

def codeFn(ecg, fs):
    nyquist = 0.5 * fs
    low_cutoff = 5
    high_cutoff = 15
    b, a = butter(1, [low_cutoff/nyquist, high_cutoff/nyquist], btype='band')
    ecg_filt = filtfilt(b, a, ecg)

    b = np.array([1, 0, -1])
    ecg_diff = np.convolve(ecg_filt, b, mode='same')
    ecg_sq = ecg_diff ** 2

    ma_len = int(0.08 * fs)
    ecg_ma = np.convolve(ecg_sq, np.ones(ma_len)/ma_len, mode='same')

    qrs_idx, _ = find_peaks(ecg_ma, distance=int(0.2 * fs), height=0.2 * np.max(ecg_ma))

    return qrs_idx

df = pd.read_csv('/content/original_ECG_1.csv', header=None)
print(df)
df = df.drop(0)
print(df)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.interpolate()
df = df.astype(float)
print(df)

ecg_signal = df.values.flatten()
fs = 250 # consider
qrs = codeFn(ecg_signal, fs)

t = np.arange(len(ecg_signal)) / fs
plt.plot(t, ecg_signal)
plt.plot(qrs/fs, ecg_signal[qrs], 'ro')
plt.xlabel('Time (s)')
plt.ylabel('ECG Amplitude')
plt.show()

qrs_df = pd.DataFrame({'QRS_Peaks': qrs})

qrs_df.to_excel('qrs_peaks.xlsx', index=False)
