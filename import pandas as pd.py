import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame with the provided data
data = {
    'Classifier': ['AdaBoost', 'CatBoost', 'Random Forest', 'Decision Tree', 'Multilayer Perceptron'],
    'Accuracy': [0.9781, 0.9614, 0.9849, 0.9779, 0.9768],
    'Precision': [0.97, 0.96, 0.98, 0.97, 0.98],
    'Recall': [0.98, 0.95, 0.98, 0.98, 0.98],
    'F1 Score': [0.97, 0.96, 0.98, 0.98, 0.98]
}

results_df = pd.DataFrame(data)

# Set the style of seaborn
sns.set(style="whitegrid")

# Create the first figure for Accuracy and Precision
plt.figure(figsize=(12, 10))

# Plot Accuracy
plt.subplot(2, 1, 1)
sns.barplot(data=results_df, x='Accuracy', y='Classifier', palette='viridis')
plt.title('Accuracy of Classifiers')
plt.xlim(0.9, 1.0)  # Set x-axis limits for better visualization

# Plot Precision
plt.subplot(2, 1, 2)
sns.barplot(data=results_df, x='Precision', y='Classifier', palette='viridis')
plt.title('Precision of Classifiers')
plt.xlim(0.9, 1.0)  # Set x-axis limits for better visualization

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  # Increase the spacing between plots
plt.show()

# Create the second figure for Recall and F1 Score
plt.figure(figsize=(12, 10))

# Plot Recall
plt.subplot(2, 1, 1)
sns.barplot(data=results_df, x='Recall', y='Classifier', palette='viridis')
plt.title('Recall of Classifiers')
plt.xlim(0.9, 1.0)  # Set x-axis limits for better visualization

# Plot F1 Score
plt.subplot(2, 1, 2)
sns.barplot(data=results_df, x='F1 Score', y='Classifier', palette='viridis')
plt.title('F1 Score of Classifiers')
plt.xlim(0.9, 1.0)  # Set x-axis limits for better visualization

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  # Increase the spacing between plots
plt.show()
