import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 6))  # Assuming each classifier corresponds to an epoch for demonstration

# Accuracy data
KDDTrain_accuracy = [0.9902, 0.9787, 0.9911, 0.9867, 0.9914]
KDDTrain_20_percent_accuracy = [0.9791, 0.9673, 0.9835, 0.9743, 0.9795]
KDDTest_accuracy = [0.7024, 0.6931, 0.7013, 0.6972, 0.7004]
KDDTest_Train_accuracy = [0.6873, 0.6908, 0.6929, 0.6872, 0.6994]

# Precision data
KDDTrain_precision = [0.98, 0.99, 0.99, 0.99, 0.98]
KDDTrain_20_percent_precision = [0.98, 0.97, 0.98, 0.97, 0.98]
KDDTest_precision = [0.60, 0.59, 0.65, 0.64, 0.65]
KDDTest_Train_precision = [0.65, 0.63, 0.64, 0.61, 0.64]

# Recall data
KDDTrain_recall = [0.97, 0.98, 0.99, 0.98, 0.98]
KDDTrain_20_percent_recall = [0.98, 0.97, 0.98, 0.97, 0.98]
KDDTest_recall = [0.69, 0.72, 0.69, 0.70, 0.70]
KDDTest_Train_recall = [0.73, 0.70, 0.71, 0.69, 0.70]

# F1-Score data
KDDTrain_f1 = [0.98, 0.99, 0.98, 0.98, 0.99]
KDDTrain_20_percent_f1 = [0.97, 0.96, 0.98, 0.97, 0.98]
KDDTest_f1 = [0.64, 0.62, 0.64, 0.62, 0.64]
KDDTest_Train_f1 = [0.63, 0.63, 0.65, 0.67, 0.63]

# Plot for Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, KDDTrain_accuracy, 'r-.', linewidth=2, label='KDDTrain Accuracy')
plt.plot(epochs, KDDTrain_20_percent_accuracy, 'g-.', linewidth=2, label='KDDTrain 20% Accuracy')
plt.plot(epochs, KDDTest_accuracy, 'b-.', linewidth=2, label='KDDTrain with KDDTest Accuracy')
plt.plot(epochs, KDDTest_Train_accuracy, 'm-.', linewidth=2, label='KDDTest with KDDTrain 20% Accuracy')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Accuracy vs Epoch', fontsize=16)
plt.legend(loc='center', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for Precision
plt.figure(figsize=(10, 5))
plt.plot(epochs, KDDTrain_precision, 'r--', linewidth=2, label='KDDTrain Precision')
plt.plot(epochs, KDDTrain_20_percent_precision, 'g--', linewidth=2, label='KDDTrain 20% Precision')
plt.plot(epochs, KDDTest_precision, 'b--', linewidth=2, label='KDDTrain with KDDTest Precision')
plt.plot(epochs, KDDTest_Train_precision, 'm--', linewidth=2, label='KDDTest with KDDTrain 20% Precision')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision vs Epoch', fontsize=16)
plt.legend(loc='center', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for Recall
plt.figure(figsize=(10, 5))
plt.plot(epochs, KDDTrain_recall, 'r:', linewidth=2, label='KDDTrain Recall')
plt.plot(epochs, KDDTrain_20_percent_recall, 'g:', linewidth=2, label='KDDTrain 20% Recall')
plt.plot(epochs, KDDTest_recall, 'b:', linewidth=2, label='KDDTrain with KDDTest Recall')
plt.plot(epochs, KDDTest_Train_recall, 'm:', linewidth=2, label='KDDTest with KDDTrain 20% Recall')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.title('Recall vs Epoch', fontsize=16)
plt.legend(loc='center', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for F1-Score
plt.figure(figsize=(10, 5))
plt.plot(epochs, KDDTrain_f1, 'r-', linewidth=2, label='KDDTrain F1-Score')
plt.plot(epochs, KDDTrain_20_percent_f1, 'g-', linewidth=2, label='KDDTrain 20% F1-Score')
plt.plot(epochs, KDDTest_f1, 'b-', linewidth=2, label='KDDTrain with KDDTest F1-Score')
plt.plot(epochs, KDDTest_Train_f1, 'm-', linewidth=2, label='KDDTest with KDDTrain 20% F1-Score')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.title('F1 Score vs Epoch', fontsize=16)
plt.legend(loc='center', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
