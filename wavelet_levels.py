import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate a sample signal
np.random.seed(0)
t = np.linspace(0, 1, 1024, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.2 * np.random.randn(1024)

# Perform multi-level wavelet decomposition using db4
coeffs = pywt.wavedec(signal, 'db4', level=4)

# Plot the original signal and detail coefficients
plt.figure(figsize=(10, 8))

# Plot the original signal
plt.subplot(5, 1, 1)
plt.plot(signal, color='darkblue', linewidth=1.5)
plt.title('Original Signal', fontsize=14, fontweight='bold', color='black')
plt.xlabel('Time', fontsize=12, fontweight='bold', color='black')
plt.ylabel('Amplitude', fontsize=12, fontweight='bold', color='black')
plt.yticks(fontsize=12, fontweight='bold', color='black')  # Set y-axis tick labels to bold and black
plt.xticks(fontsize=12, fontweight='bold', color='black')
plt.tick_params(axis='both', labelsize=12, labelcolor='black', width=2, length=6)

# Plot the detail coefficients at each level
for i, c in enumerate(coeffs[1:]):
    plt.subplot(5, 1, i + 2)
    plt.plot(c, color='darkblue', linewidth=1.5)
    plt.title(f'Wavelet Coefficients Level {i + 1}', fontsize=14, fontweight='bold', color='black')
    plt.xlabel('Time', fontsize=12, fontweight='bold', color='black')
    plt.ylabel('Amplitude', fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')  # Set y-axis tick labels to bold and black
    plt.xticks(fontsize=12, fontweight='bold', color='black')
    plt.tick_params(axis='both', labelsize=12, labelcolor='black', width=2, length=6)

plt.tight_layout()
plt.show()
