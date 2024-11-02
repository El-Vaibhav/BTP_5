import numpy as np

# Define the metrics as lists
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

# Function to calculate average of metrics across the datasets
def calculate_average(metrics):
    return [np.mean([metrics[i][j] for i in range(len(metrics))]) for j in range(len(metrics[0]))]

# Calculate averages for each metric
average_accuracy = calculate_average([KDDTrain_accuracy, KDDTrain_20_percent_accuracy, KDDTest_accuracy, KDDTest_Train_accuracy])
average_precision = calculate_average([KDDTrain_precision, KDDTrain_20_percent_precision, KDDTest_precision, KDDTest_Train_precision])
average_recall = calculate_average([KDDTrain_recall, KDDTrain_20_percent_recall, KDDTest_recall, KDDTest_Train_recall])
average_f1 = calculate_average([KDDTrain_f1, KDDTrain_20_percent_f1, KDDTest_f1, KDDTest_Train_f1])

# Display the averages
print("Average Metrics:")
print(f"Average Accuracy: {average_accuracy}")
print(f"Average Precision: {average_precision}")
print(f"Average Recall: {average_recall}")
print(f"Average F1 Score: {average_f1}")

# Find best two from each case
def best_two(metrics):
    return sorted(enumerate(metrics), key=lambda x: x[1], reverse=True)[:2]

print("\nBest 2 Models for Each Metric:")
print(f"Best 2 Accuracies: {best_two(average_accuracy)}")
print(f"Best 2 Precisions: {best_two(average_precision)}")
print(f"Best 2 Recalls: {best_two(average_recall)}")
print(f"Best 2 F1 Scores: {best_two(average_f1)}")
