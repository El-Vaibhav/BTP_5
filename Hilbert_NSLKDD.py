import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier

# Read the dataset
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+.txt", header=None)

# Drop non-numeric columns, keep the last column for labels
df_numeric = df.drop(columns=[1, 2, 3, df.columns[-2]])

# Convert remaining columns to float
df_numeric = df_numeric.astype(float)

# Extract labels
labels = df[df.columns[-2]]

# Convert numeric data to numpy array
data_numeric = df_numeric.values

# Apply Hilbert Transform
hilbert_amplitude = []

for row in data_numeric:
    analytic_signal = hilbert(row)
    amplitude_envelope = np.abs(analytic_signal)
    hilbert_amplitude.append(amplitude_envelope)

# Convert to numpy array
hilbert_amplitude = np.array(hilbert_amplitude)

# Flatten the Hilbert amplitude for each sample to use as features
X = hilbert_amplitude.reshape(hilbert_amplitude.shape[0], -1)
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Confusion Matrix for {name}:")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"Confusion Matrix - {name}")
    plt.show()
    
    # Print accuracy and classification report
    print(f"Accuracy of {name}: {accuracy:.7f}")
    print(classification_report(y_test, y_pred, zero_division=0))

# Plotting the Hilbert amplitude of each row with different colors
colors = {'normal.': 'blue', 'smurf.': 'black', 'neptune.': 'red', 'back': 'brown', 'pod': 'pink', 'teardrop': 'yellow', 'buffer_overflow': 'green', 'warezclient': 'lightblue', 'ipsweep': 'darkred', 'portsweep': 'aqua', 'satan': 'blueviolet'}

plt.figure(figsize=(14, 6))
for i in range(len(df)):
    if labels[i] in colors:
        plt.plot(np.arange(hilbert_amplitude.shape[1]), hilbert_amplitude[i], color=colors[labels[i]], linewidth=2.5)

legend_entries = [
    plt.Line2D([0], [0], color='blue', label='normal', linewidth=2.5),
    plt.Line2D([0], [0], color='black', label='smurf (DOS attack)', linewidth=2.5),
    plt.Line2D([0], [0], color='red', label='neptune (DOS attack)', linewidth=2.5),
    plt.Line2D([0], [0], color='brown', label='back (DOS attack)', linewidth=2.5),
    plt.Line2D([0], [0], color='pink', label='pod (DOS attack)', linewidth=2.5),
    plt.Line2D([0], [0], color='yellow', label='teardrop (DOS attack)', linewidth=2.5),
    plt.Line2D([0], [0], color='green', label='buffer_overflow (U2R attack)', linewidth=2.5),
    plt.Line2D([0], [0], color='lightblue', label='warezclient (R2L attack)', linewidth=2.5),
    plt.Line2D([0], [0], color='darkred', label='ipsweep (Probe attack)', linewidth=2.5),
    plt.Line2D([0], [0], color='aqua', label='portsweep (Probe attack)', linewidth=2.5),
    plt.Line2D([0], [0], color='blueviolet', label='satan (Probe attack)', linewidth=2.5),
]

plt.legend(handles=legend_entries, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
plt.title("Hilbert Transform Amplitude")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.yscale('log')
plt.tight_layout()
plt.show()