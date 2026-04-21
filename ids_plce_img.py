import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.signal

def compute_wavelet_coherence(x, y, wavelet='morl', scales=np.arange(1, 128)):
    coef_x, _ = pywt.cwt(x, scales, wavelet)
    coef_y, _ = pywt.cwt(y, scales, wavelet)

    Sxy = np.abs(coef_x * np.conj(coef_y))
    Sxx = np.abs(coef_x) ** 2
    Syy = np.abs(coef_y) ** 2

    coherence = Sxy ** 2 / (Sxx * Syy + 1e-10)
    return coherence

# Example dummy signals
x = np.sin(np.linspace(0, 20 * np.pi, 1024))
y = x + np.random.normal(0, 0.3, 1024)

coherence = compute_wavelet_coherence(x, y)

plt.figure(figsize=(10, 5))
plt.imshow(coherence, extent=[0, len(x), 1, 128], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Wavelet Coherence')
plt.title('Wavelet Coherence Between Node x and y', fontsize=14, fontweight='bold', color='black')
plt.xlabel('Time', fontsize=12, fontweight='bold', color='black')
plt.ylabel('Scale', fontsize=12, fontweight='bold', color='black')
plt.tick_params(axis='both', labelsize=12, labelcolor='black', width=2, length=6)
plt.yticks(fontsize=12, fontweight='bold', color='black')  # Set y-axis tick labels to bold and black
plt.xticks(fontsize=12, fontweight='bold', color='black')  # Set y-axis tick labels to bold and black
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the dimensions
time = np.linspace(0, 1000, 1000)
scale = np.arange(1, 128)

# Create synthetic coherence data
coherence = np.ones((len(scale), len(time)))

# Introduce an anomaly (drop in coherence)
anomaly_start = 400
anomaly_end = 600
coherence[:, anomaly_start:anomaly_end] = 0.2  # Drop coherence to 0.2

# Plot the heatmap
plt.figure(figsize=(10, 6))
plt.imshow(coherence, aspect='auto', cmap='viridis', extent=[time.min(), time.max(), scale.min(), scale.max()])
plt.colorbar(label='Wavelet Coherence')
plt.title('Wavelet Coherence with Anomaly Indication', fontsize=14, fontweight='bold', color='black')
plt.xlabel('Time', fontsize=12, fontweight='bold', color='black')
plt.ylabel('Scale', fontsize=12, fontweight='bold', color='black')
plt.tick_params(axis='both', labelsize=12, labelcolor='black', width=2, length=6)
plt.yticks(fontsize=12, fontweight='bold', color='black')  # Set y-axis tick labels to bold and black
plt.xticks(fontsize=12, fontweight='bold', color='black')  # Set y-axis tick labels to bold and black
plt.axvline(x=anomaly_start, color='red', linestyle='--', label='Anomaly Start')
plt.axvline(x=anomaly_end, color='red', linestyle='--', label='Anomaly End')
plt.legend()
plt.show()
