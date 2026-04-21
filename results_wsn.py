import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
classifiers = ['AdaBoost', 'CatBoost', 'Random Forest', 'Decision Tree', 'Multilayer Perceptron']
accuracy = [0.9781, 0.9614, 0.9849, 0.9779, 0.9768]
precision = [0.97, 0.96, 0.98, 0.97, 0.98]

# Create a DataFrame
df = pd.DataFrame({
    'Classifier': classifiers,
    'Accuracy': accuracy,
    'Precision': precision
})

# Set the style
sns.set(style="whitegrid")
palette = sns.color_palette("viridis", len(df))

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))

# Accuracy Plot
sns.barplot(x='Accuracy', y='Classifier', data=df, palette=palette, ax=axes[0])
axes[0].set_title('Accuracy of Classifiers', fontsize=14, weight='bold', color='black')
axes[0].set_xlim(0.90, 1.00)
axes[0].tick_params(axis='both', labelsize=11, labelcolor='black', width=2, length=6)
axes[0].set_xlabel("Accuracy", fontsize=12, weight='bold', color='black')
axes[0].set_ylabel("Classifier", fontsize=12, fontweight='bold', color='black')
axes[0].set_xticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
axes[0].set_xticklabels(['0.90', '0.92', '0.94', '0.96', '0.98', '1.00'], fontsize=12, fontweight='bold', color='black')
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=12, fontweight='bold', color='black')  # Set y-axis tick labels to bold and black

# Precision Plot
sns.barplot(x='Precision', y='Classifier', data=df, palette=palette, ax=axes[1])
axes[1].set_title('Precision of Classifiers', fontsize=14, weight='bold', color='black')
axes[1].set_xlim(0.90, 1.00)
axes[1].tick_params(axis='both', labelsize=11, labelcolor='black', width=2, length=6)
axes[1].set_xlabel("Precision", fontsize=12, weight='bold', color='black')
axes[1].set_ylabel("Classifier", fontsize=12, fontweight='bold', color='black')
axes[1].set_xticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
axes[1].set_xticklabels(['0.90', '0.92', '0.94', '0.96', '0.98', '1.00'], fontsize=12, fontweight='bold', color='black')
axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=12, fontweight='bold', color='black')  # Set y-axis tick labels to bold and black

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
classifiers = ['AdaBoost', 'CatBoost', 'Random Forest', 'Decision Tree', 'Multilayer Perceptron']
precision = [0.98, 0.95, 0.98, 0.98, 0.98]
recall = [0.97, 0.96, 0.98, 0.98, 0.98]

# Create a DataFrame
df = pd.DataFrame({
    'Classifier': classifiers,
    'Precision': precision,
    'Recall': recall
})

# Set the style
sns.set(style="whitegrid")
palette = sns.color_palette("viridis", len(df))

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))

# Precision Plot
sns.barplot(x='Precision', y='Classifier', data=df, palette=palette, ax=axes[0])
axes[0].set_title('Precision of Classifiers', fontsize=14, weight='bold', color='black')
axes[0].set_xlim(0.90, 1.00)
axes[0].tick_params(axis='both', labelsize=11, labelcolor='black', width=2, length=6)
axes[0].set_xlabel("Precision", fontsize=12, weight='bold', color='black')
axes[0].set_ylabel("Classifier", fontsize=12, fontweight='bold', color='black')
axes[0].set_xticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
axes[0].set_xticklabels(['0.90', '0.92', '0.94', '0.96', '0.98', '1.00'], fontsize=12, fontweight='bold', color='black')
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=12, fontweight='bold', color='black')  # Set y-axis tick labels to bold and black

# Recall Plot
sns.barplot(x='Recall', y='Classifier', data=df, palette=palette, ax=axes[1])
axes[1].set_title('Recall of Classifiers', fontsize=14, weight='bold', color='black')
axes[1].set_xlim(0.90, 1.00)
axes[1].tick_params(axis='both', labelsize=11, labelcolor='black', width=2, length=6)
axes[1].set_xlabel("Recall", fontsize=12, weight='bold', color='black')
axes[1].set_ylabel("Classifier", fontsize=12, fontweight='bold', color='black')
axes[1].set_xticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
axes[1].set_xticklabels(['0.90', '0.92', '0.94', '0.96', '0.98', '1.00'], fontsize=12, fontweight='bold', color='black')
axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=12, fontweight='bold', color='black')  # Set y-axis tick labels to bold and black

plt.tight_layout()
plt.show()

