import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # for smoothing

# Simulated time axis and coherence arrays
t = np.linspace(0, 10, 500)  # 500 time points

# Suppose these are your average coherence over time (replace with real data)
avg_coh_0 = np.random.rand(500)*0.2 + 0.6  # simulate Node 0 with a dip
avg_coh_1 = np.random.rand(500)*0.2 + 0.5  # simulate Node 1 with dip

# Introduce a coherence drop during anomaly window
avg_coh_0[200:300] -= 0.4
avg_coh_1[200:300] -= 0.3

# Smooth
avg_coh_0 = savgol_filter(avg_coh_0, 31, 3)
avg_coh_1 = savgol_filter(avg_coh_1, 31, 3)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t, avg_coh_0, label='Node 0', color='red')
plt.plot(t, avg_coh_1, label='Node 1', color='blue')
plt.axvspan(4, 6, color='gray', alpha=0.3, label='Anomaly Window')  # highlight anomaly

plt.xlabel("Time")
plt.ylabel("Mean Coherence with Other Nodes")
plt.title("Coherence Drop Over Time (Node 0 & 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
