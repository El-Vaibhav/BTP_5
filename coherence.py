import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import coherence
from codecarbon import EmissionsTracker
import time

# ---------- PARAMETERS ----------
np.random.seed(42)
n_nodes = 10           # Number of nodes
duration = 60                # Simulated time duration in seconds
sampling_rate = 10           # Frequency in Hz (samples per second)
t = np.linspace(0, duration, duration * sampling_rate)

# Simulated network traffic with anomalies
def generate_traffic(normal=True):
    base = np.sin(2 * np.pi * 1 * t) + np.random.normal(0, 0.5, len(t))  # Low frequency
    if not normal:
        spike = np.zeros_like(t)
        spike[200:500] = 7 * np.sin(2 * np.pi * 3 * t[200:500]) + np.random.normal(0, 1, 300)
        base += spike
    high_freq = np.sin(2 * np.pi * 5 * t)  # High frequency signal
    base += high_freq * np.random.normal(0, 0.3, len(t))  # Add variability with high frequency
    return base

# Function to compute wavelet coherence
def wavelet_coherence(sig1, sig2, wavelet='db4'):
    coeffs1 = pywt.wavedec(sig1, wavelet, level=4)
    coeffs2 = pywt.wavedec(sig2, wavelet, level=4)
    
    # Use a different level of wavelet coefficients for more variability
    f, Cxy = coherence(coeffs1[1], coeffs2[1], fs=sampling_rate)
    return np.mean(Cxy)


# ------------------- SIMULATION -------------------
tracker = EmissionsTracker()
tracker.start()
start_time = time.time()

# Simulate traffic at each node
traffic_signals = [generate_traffic(normal=True if i % 2 == 0 else False) for i in range(n_nodes)]

# Compute pairwise coherence
coherence_matrix = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        coh = wavelet_coherence(traffic_signals[i], traffic_signals[j])
        coherence_matrix[i][j] = coherence_matrix[j][i] = coh
        # Print the coherence for each pair of nodes
        print(f"Coherence between Node {i} and Node {j}: {coh:.4f}")

# IDS placement: choose nodes with lowest average coherence
avg_coh = np.mean(coherence_matrix, axis=1)
ids_nodes = np.argsort(avg_coh)[:2]  # Top 2 low-coherence nodes

# Display results
end_time = time.time()
emissions = tracker.stop()

print("\nCoherence Matrix:\n", coherence_matrix)
print("\nAverage Coherence per Node:", avg_coh)
print("\nSelected IDS Nodes (low coherence):", ids_nodes)
print("\nExecution Time: %.2f seconds" % (end_time - start_time))
print("Estimated Energy (kWh):", emissions)

# ------------ PLOT: Traffic with IDS Highlighted ------------
plt.figure(figsize=(12, 6))
for i, signal in enumerate(traffic_signals):
    if i in ids_nodes:
        plt.plot(t, signal + i * 10, label=f'Node {i} (IDS)', linewidth=2.5, color='red')
    else:
        plt.plot(t, signal + i * 10, label=f'Node {i}', linestyle='--', alpha=0.6)

plt.xticks(color='black', fontweight='bold')
plt.yticks(color='black', fontweight='bold')

plt.title("Simulated Network Traffic with IDS Node Detection",color='black', fontweight='bold')
plt.xlabel("Time (s)",color='black', fontweight='bold')
plt.ylabel("Signal Amplitude + Offset",color='black', fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.02), 
           prop={'weight': 'bold', 'size': 10}, labelcolor='black')
plt.tight_layout()
plt.grid(True)
plt.show()
