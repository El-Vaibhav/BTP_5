import numpy as np
import matplotlib.pyplot as plt

# Datasets
datasets = ["KDDTrain+", "KDDTrain+20%", "KDDTrain+ with KDDTest+", "KDDTrain+20% with KDDTest+"]
x_positions = np.arange(len(datasets))  # Better spacing

# Classifier Metrics
metrics = {
    "Accuracy": {
        "Ada Boost": [0.9902, 0.9791, 0.7024, 0.6873],
        "Cat Boost": [0.9787, 0.9673, 0.6931, 0.6908],
        "Random Forest": [0.9911, 0.9835, 0.7013, 0.6929],
        "Decision Tree": [0.9867, 0.9743, 0.6972, 0.6872],
        "MLP": [0.9914, 0.9795, 0.7004, 0.6994]
    },
    "Precision": {
        "Ada Boost": [0.98, 0.98, 0.60, 0.65],
        "Cat Boost": [0.99, 0.97, 0.59, 0.63],
        "Random Forest": [0.99, 0.98, 0.65, 0.64],
        "Decision Tree": [0.99, 0.97, 0.64, 0.61],
        "MLP": [0.98, 0.98, 0.65, 0.64]
    },
    "Recall": {
        "Ada Boost": [0.97, 0.98, 0.69, 0.73],
        "Cat Boost": [0.98, 0.97, 0.72, 0.70],
        "Random Forest": [0.99, 0.98, 0.69, 0.71],
        "Decision Tree": [0.98, 0.97, 0.70, 0.69],
        "MLP": [0.98, 0.98, 0.70, 0.70]
    },
    "F1 Score": {
        "Ada Boost": [0.98, 0.97, 0.64, 0.63],
        "Cat Boost": [0.99, 0.96, 0.62, 0.63],
        "Random Forest": [0.98, 0.98, 0.64, 0.65],
        "Decision Tree": [0.98, 0.97, 0.62, 0.67],
        "MLP": [0.99, 0.98, 0.64, 0.67]
    }
}

# Define line styles and colors
line_styles = ['--', '--', '--', '--']  # Different line styles for variety
colors = ['red', 'blue', 'green', 'purple']  # Distinct colors

plt.figure(figsize=(12, 7))

# Plot each metric with a different line style
for idx, (metric, classifiers) in enumerate(metrics.items()):
    avg_values = np.mean(list(classifiers.values()), axis=0)  # Compute average per dataset
    plt.plot(x_positions, avg_values, linestyle=line_styles[idx % len(line_styles)], 
             color=colors[idx % len(colors)], linewidth=3, label=metric)

# Formatting the plot
plt.xticks(x_positions, datasets, rotation=15, fontsize=12,color="black",weight="bold")  # Improve readability
plt.yticks(fontsize=12,weight="bold",color="black")

plt.xlabel("Datasets", fontsize=14, fontweight="bold")
plt.ylabel("Score", fontsize=14, fontweight="bold")
plt.title("Performance Comparison of Classifiers", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0.5, 1.0)  # Keep y-axis consistent

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Classifiers
classifiers = ["Ada Boost", "Cat Boost", "Random Forest", "Decision Tree", "MLP"]

# Performance Metrics
accuracy = [0.9781, 0.9614, 0.9849, 0.9779, 0.9768]
precision = [0.97, 0.96, 0.98, 0.97, 0.98]
recall = [0.98, 0.95, 0.98, 0.98, 0.98]
f1_score = [0.97, 0.96, 0.98, 0.98, 0.98]

# Define line styles, markers, and colors
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^']
colors = ['red', 'blue', 'green', 'purple']

plt.figure(figsize=(10, 6))

# Plot each metric with different styles
plt.plot(classifiers, accuracy, linestyle=line_styles[0], marker=markers[0], color=colors[0], linewidth=2, label="Accuracy")
plt.plot(classifiers, precision, linestyle=line_styles[1], marker=markers[1], color=colors[1], linewidth=2, label="Precision")
plt.plot(classifiers, recall, linestyle=line_styles[2], marker=markers[2], color=colors[2], linewidth=2, label="Recall")
plt.plot(classifiers, f1_score, linestyle=line_styles[3], marker=markers[3], color=colors[3], linewidth=2, label="F1 Score")

# Formatting
plt.xlabel("Classifiers", fontsize=14, fontweight="bold")
plt.ylabel("Score", fontsize=14, fontweight="bold")
plt.title("Performance Metrics of Classifiers on WSN-DS Dataset", fontsize=16, fontweight="bold")
plt.xticks(rotation=15, fontsize=12,weight="bold",color="black")  # Rotate labels for readability
plt.yticks(fontsize=12,weight="bold",color="black")
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0.9, 1.0)  # Keep the range visually appealing

# Show plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Classifiers
classifiers = ["Ada Boost", "Cat Boost", "Random Forest", "Decision Tree", "MLP"]

# Performance Metrics
accuracy = [0.9987, 0.9854, 0.9990, 0.9977, 0.9981]
precision = [0.99, 0.98, 0.99, 0.98, 0.99]
recall = [0.99, 0.98, 0.99, 0.98, 0.99]
f1_score = [0.99, 0.99, 0.98, 0.99, 0.98]

# Define line styles, markers, and colors
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^']
colors = ['red', 'blue', 'green', 'purple']

plt.figure(figsize=(10, 6))

# Plot each metric with different styles
plt.plot(classifiers, accuracy, linestyle=line_styles[0], marker=markers[0], color=colors[0], linewidth=2, label="Accuracy")
plt.plot(classifiers, precision, linestyle=line_styles[1], marker=markers[1], color=colors[1], linewidth=2, label="Precision")
plt.plot(classifiers, recall, linestyle=line_styles[2], marker=markers[2], color=colors[2], linewidth=2, label="Recall")
plt.plot(classifiers, f1_score, linestyle=line_styles[3], marker=markers[3], color=colors[3], linewidth=2, label="F1 Score")

# Formatting
plt.xlabel("Classifiers", fontsize=14, fontweight="bold")
plt.ylabel("Score", fontsize=14, fontweight="bold")
plt.title("Performance Metrics of Classifiers on NF-TON-IoT Dataset", fontsize=16, fontweight="bold")
plt.xticks(rotation=15, fontsize=12,weight="bold",color="black")  # Rotate labels for readability
plt.yticks(fontsize=12,weight="bold",color="black")
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0.95, 1.0)  # Keep the range visually appealing

# Show plot
plt.show()
