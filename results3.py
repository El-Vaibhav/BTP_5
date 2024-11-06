import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame with the provided data
data = {
    'Classifier': ['AdaBoost', 'CatBoost', 'Random Forest', 'Decision Tree', 'Multilayer Perceptron'],
    'Accuracy': [0.9987, 0.9854, 0.9990, 0.9977, 0.9981],
    'Precision': [0.99, 0.98, 0.99, 0.98, 0.99],
    'Recall': [0.99, 0.98, 0.99, 0.98, 0.99],
    'F1 Score': [0.99, 0.99, 0.98, 0.99, 0.98]
}

results_df = pd.DataFrame(data)

# Set the style of seaborn
sns.set(style="whitegrid")

# Define x-axis tick positions including 0.99
x_ticks = [0.90, 0.95, 0.99, 1.0]

# Create the first figure for Accuracy and Precision
plt.figure(figsize=(12, 10))

# Plot Accuracy with a 'coolwarm' palette
plt.subplot(2, 1, 1)
sns.barplot(data=results_df, x='Accuracy', y='Classifier', palette='coolwarm')
plt.title('Accuracy of Classifiers')
plt.xlim(0.90, 1.0)
plt.xticks(x_ticks)  # Set specific x-axis ticks

# Plot Precision with a 'crest' palette
plt.subplot(2, 1, 2)
sns.barplot(data=results_df, x='Precision', y='Classifier', palette='crest')
plt.title('Precision of Classifiers')
plt.xlim(0.90, 1.0)
plt.xticks(x_ticks)  # Set specific x-axis ticks

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()

# Create the second figure for Recall and F1 Score
plt.figure(figsize=(12, 10))

# Plot Recall with a 'flare' palette
plt.subplot(2, 1, 1)
sns.barplot(data=results_df, x='Recall', y='Classifier', palette='flare')
plt.title('Recall of Classifiers')
plt.xlim(0.90, 1.0)
plt.xticks(x_ticks)  # Set specific x-axis ticks

# Plot F1 Score with a 'mako' palette
plt.subplot(2, 1, 2)
sns.barplot(data=results_df, x='F1 Score', y='Classifier', palette='mako')
plt.title('F1 Score of Classifiers')
plt.xlim(0.90, 1.0)
plt.xticks(x_ticks)  # Set specific x-axis ticks

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()
