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

# Define x-axis tick positions
x_ticks = [0.90, 0.95, 0.99, 1.0]

# Set global text color to black
plt.rcParams.update({'text.color': 'black',
                     'axes.labelcolor': 'black',
                     'xtick.color': 'black',
                     'ytick.color': 'black'})

# Create the first figure for Accuracy and Precision
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
sns.barplot(data=results_df, x='Accuracy', y='Classifier', palette='coolwarm')
plt.title('Accuracy of Classifiers', color='black',fontweight='bold')
plt.xlabel('Accuracy', color='black',fontweight='bold')
plt.ylabel('Classifier', color='black',fontweight='bold')
plt.xlim(0.90, 1.0)
plt.xticks(x_ticks, color='black',fontweight='bold')
plt.yticks(color='black',fontweight='bold')

plt.subplot(2, 1, 2)
sns.barplot(data=results_df, x='Precision', y='Classifier', palette='crest')
plt.title('Precision of Classifiers', color='black',fontweight='bold')
plt.xlabel('Precision', color='black',fontweight='bold')
plt.ylabel('Classifier', color='black',fontweight='bold')
plt.xlim(0.90, 1.0)
plt.xticks(x_ticks, color='black',fontweight='bold')
plt.yticks(color='black',fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()

# Create the second figure for Recall and F1 Score
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
sns.barplot(data=results_df, x='Recall', y='Classifier', palette='flare')
plt.title('Recall of Classifiers', color='black',fontweight='bold')
plt.xlabel('Recall', color='black',fontweight='bold')
plt.ylabel('Classifier', color='black',fontweight='bold')
plt.xlim(0.90, 1.0)
plt.xticks(x_ticks, color='black',fontweight='bold')
plt.yticks(color='black',fontweight='bold')

plt.subplot(2, 1, 2)
sns.barplot(data=results_df, x='F1 Score', y='Classifier', palette='mako')
plt.title('F1 Score of Classifiers', color='black',fontweight='bold')
plt.xlabel('F1 Score', color='black',fontweight='bold')
plt.ylabel('Classifier', color='black',fontweight='bold')
plt.xlim(0.90, 1.0)
plt.xticks(x_ticks, color='black',fontweight='bold')
plt.yticks(color='black',fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()
