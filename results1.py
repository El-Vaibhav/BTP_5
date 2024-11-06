import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Classifiers
classifiers = ['AdaBoost', 'CatBoost', 'Random Forest', 'Decision Tree', 'MLP']

# Adjusted Performance metrics data
KDDTrain_accuracy = [0.9902, 0.9787, 0.9911, 0.9867, 0.9914]
KDDTrain_20_percent_accuracy = [0.9791, 0.9673, 0.9835, 0.9743, 0.9795]
KDDTest_accuracy = [0.7024, 0.6931, 0.7013, 0.6972, 0.7004]
KDDTest_Train_accuracy = [0.6873, 0.6908, 0.6929, 0.6872, 0.6994]

KDDTrain_precision = [0.98, 0.99, 0.99, 0.99, 0.98]
KDDTrain_20_percent_precision = [0.98, 0.97, 0.98, 0.97, 0.98]
KDDTest_precision = [0.60, 0.59, 0.65, 0.64, 0.65]
KDDTest_Train_precision = [0.65, 0.63, 0.64, 0.61, 0.64]

KDDTrain_recall = [0.97, 0.98, 0.99, 0.98, 0.98]
KDDTrain_20_percent_recall = [0.98, 0.97, 0.98, 0.97, 0.98]
KDDTest_recall = [0.69, 0.72, 0.69, 0.70, 0.70]
KDDTest_Train_recall = [0.73, 0.70, 0.71, 0.69, 0.70]

KDDTrain_f1 = [0.98, 0.99, 0.98, 0.98, 0.99]
KDDTrain_20_percent_f1 = [0.97, 0.96, 0.98, 0.97, 0.98]
KDDTest_f1 = [0.64, 0.62, 0.64, 0.62, 0.64]
KDDTest_Train_f1 = [0.63, 0.63, 0.65, 0.67, 0.63]

# Error margins (example placeholder, since actual error margins were not provided)
KDDTrain_errors = [0.005, 0.005, 0.005, 0.005, 0.005]
KDDTrain_20_percent_errors = [0.005, 0.005, 0.005, 0.005, 0.005]
KDDTest_errors = [0.01, 0.01, 0.01, 0.01, 0.01]
KDDTest_Train_errors = [0.01, 0.01, 0.01, 0.01, 0.01]

# Organize the metrics into a nested list
metrics_data = [
    [KDDTrain_accuracy, KDDTrain_20_percent_accuracy, KDDTest_accuracy, KDDTest_Train_accuracy],
    [KDDTrain_precision, KDDTrain_20_percent_precision, KDDTest_precision, KDDTest_Train_precision],
    [KDDTrain_recall, KDDTrain_20_percent_recall, KDDTest_recall, KDDTest_Train_recall],
    [KDDTrain_f1, KDDTrain_20_percent_f1, KDDTest_f1, KDDTest_Train_f1]
]

# Define the labels and plot each metric without error bars
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
n_classifiers = len(classifiers)
bar_width = 0.16
index = np.arange(n_classifiers)

viridis_palette = sns.color_palette("viridis", 8)

for i, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    
    # Plot each dataset's bars without error bars
    plt.bar(index, metrics_data[i][0], bar_width, color=viridis_palette[0], label='KDDTrain')
    plt.bar(index + bar_width, metrics_data[i][1], bar_width,color=viridis_palette[1], label='KDDTrain_20%')
    plt.bar(index + 2 * bar_width, metrics_data[i][2], bar_width, color=viridis_palette[2], label='KDDTrain with KDDTest')
    plt.bar(index + 3 * bar_width, metrics_data[i][3], bar_width, color=viridis_palette[3], label='KDDTrain_20% with KDDTest')

    # Add labels and title
    plt.xlabel('Classifiers')
    plt.ylabel(metric)
    plt.title(f'Comparison of {metric} on Different Classifiers')
    
    # Add ticks on x-axis
    plt.xticks(index + 1.5 * bar_width, classifiers)
    
    # Add legend
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()


#  Calculate average metric values for each classifier across different datasets
average_accuracy = np.mean([KDDTrain_accuracy, KDDTrain_20_percent_accuracy, KDDTest_accuracy, KDDTest_Train_accuracy], axis=0)
average_precision = np.mean([KDDTrain_precision, KDDTrain_20_percent_precision, KDDTest_precision, KDDTest_Train_precision], axis=0)
average_recall = np.mean([KDDTrain_recall, KDDTrain_20_percent_recall, KDDTest_recall, KDDTest_Train_recall], axis=0)
average_f1 = np.mean([KDDTrain_f1, KDDTrain_20_percent_f1, KDDTest_f1, KDDTest_Train_f1], axis=0)

print(average_accuracy,average_precision,average_recall,average_f1)


# Calculate average errors for each metric (1 - average metric value)
average_accuracy_error = 1 - average_accuracy
average_precision_error = 1 - average_precision
average_recall_error = 1 - average_recall
average_f1_error = 1 - average_f1

# Data for plotting
metrics = ['Accuracy Error', 'Precision Error', 'Recall Error', 'F1 Score Error']
errors = [average_accuracy_error, average_precision_error, average_recall_error, average_f1_error]

# Plotting
x = np.arange(len(classifiers))  # the label locations
bar_height = 0.2  # height of the bars
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each metric error as a group of horizontal bars
for i, (metric, error) in enumerate(zip(metrics, errors)):
  ax.barh(x + i * bar_height, error, bar_height, color=viridis_palette[i+2], label=metric)


# Customizations
ax.set_xlabel('Average Error')
ax.set_ylabel('Classifiers')
ax.set_title('Comparison of Average Errors across Classifiers')
ax.set_yticks(x + bar_height * 1.5)
ax.set_yticklabels(classifiers)
ax.legend()

plt.tight_layout()
plt.show()

# Error margins (we'll use the same placeholder values)
accuracy_errors = [0.01] * len(classifiers)
precision_errors = [0.01] * len(classifiers)
recall_errors = [0.01] * len(classifiers)
f1_errors = [0.01] * len(classifiers)

# Data for plotting
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [average_accuracy, average_precision, average_recall, average_f1]
errors = [accuracy_errors, precision_errors, recall_errors, f1_errors]

# Plotting
x = np.arange(len(classifiers))  # the label locations
bar_width = 0.2  # width of the bars
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each metric as a group of bars with error bars
for i, (metric, value, error) in enumerate(zip(metrics, values, errors)):
    ax.bar(x + i * bar_width, value, bar_width, yerr=error,palette='flare', label=metric, capsize=5)

# Customizations
ax.set_xlabel('Classifiers')
ax.set_ylabel('Average Score')
ax.set_title('Comparison of Average Metrics across Classifiers')
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(classifiers)
ax.legend()

plt.tight_layout()
plt.show()