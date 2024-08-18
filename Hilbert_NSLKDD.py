import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Function to preprocess data and apply Hilbert Transform
def preprocess_and_transform(filepath):
    df = pd.read_csv(filepath, header=None)

    # Drop non-numeric columns, keep the last column for labels
    df_numeric = df.drop(columns=[1, 2, 3, df.columns[-2]])
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

    return X, y

# Load and preprocess training data
X_train, y_train = preprocess_and_transform("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+.txt")

# Load and preprocess test data
X_test, y_test = preprocess_and_transform("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTest+.txt")

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")
    # Train the classifier
    clf.fit(X_train, y_train)

    # Evaluate on validation data
    y_val_pred = clf.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    cm_val = confusion_matrix(y_val, y_val_pred)
    
    print(f"Validation Confusion Matrix for {name}:")
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
    plt.title(f"Validation Confusion Matrix - {name}")
    plt.show()
    
    print(f"Validation Accuracy of {name}: {accuracy_val:.7f}")
    print(classification_report(y_val, y_val_pred, zero_division=0))
    
    # Evaluate on test data
    y_test_pred = clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    print(f"Test Confusion Matrix for {name}:")
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Test Confusion Matrix - {name}")
    plt.show()
    
    # Print accuracy and classification report
    print(f"Test Accuracy of {name}: {accuracy_test:.7f}")
    print(classification_report(y_test, y_test_pred, zero_division=0))

# Plotting the Hilbert amplitude of each row with different colors for training data
colors = {'normal.': 'blue', 'smurf.': 'black', 'neptune.': 'red', 'back': 'brown', 'pod': 'pink', 'teardrop': 'yellow', 'buffer_overflow': 'green', 'warezclient': 'lightblue', 'ipsweep': 'darkred', 'portsweep': 'aqua', 'satan': 'blueviolet'}

plt.figure(figsize=(14, 6))
for i in range(len(X_train)):
    if y_train.iloc[i] in colors:
        plt.plot(np.arange(X_train.shape[1]), X_train[i], color=colors[y_train.iloc[i]], linewidth=2.5)

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
plt.title("Hilbert Transform Amplitude (Training Data)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.yscale('log')
plt.tight_layout()
plt.show()

# Plotting the Hilbert amplitude of each row with different colors for test data
plt.figure(figsize=(14, 6))
for i in range(len(X_test)):
    if y_test.iloc[i] in colors:
        plt.plot(np.arange(X_test.shape[1]), X_test[i], color=colors[y_test.iloc[i]], linewidth=2.5)

plt.legend(handles=legend_entries, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
plt.title("Hilbert Transform Amplitude (Test Data)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.yscale('log')
plt.tight_layout()
plt.show()
