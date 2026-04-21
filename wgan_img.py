import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['NSL-KDD', 'WSN-DS', 'NF-TON-IoT']
percentage_added = [25, 30, 20]

# Colors for the bars
colors = ['#01108B', '#8B1100', '#016410']  # Dark blue, dark red, dark green

# Create the bar plot
plt.figure(figsize=(8, 6))
bars = plt.bar(datasets, percentage_added, color=colors)

# Customize the plot
plt.title('Percentage of Artificial Data Added through WGAN', fontsize=14, fontweight='bold', color='black')
plt.xlabel('Datasets', fontsize=12, fontweight='bold', color='black')
plt.ylabel('Percentage of Data Added (%)', fontsize=12, fontweight='bold', color='black')
plt.xticks(fontsize=12, fontweight='bold', color='black')
plt.yticks(fontsize=12, fontweight='bold', color='black')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

plt.tight_layout()
plt.show()
