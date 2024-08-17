import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Read the dataset
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\data.txt", header=None)

# Drop non-numeric columns, keep the last column for labels
df_numeric = df.drop(columns=[1, 2, 3, df.columns[-1]])

# Convert remaining columns to float
df_numeric = df_numeric.astype(float)

# Apply windowing to reduce spectral leakage
window = windows.hamming(df_numeric.shape[1])
data_numeric = df_numeric.values * window

# Extract labels
labels = df[df.columns[-1]]

# Convert numeric data to numpy array
data_numeric = np.array(data_numeric)

# Apply FFT along each row (axis=1)
fft_data = np.fft.fft(data_numeric, axis=1)

# Take the magnitude of the FFT results
fft_magnitude = np.abs(fft_data)

# Define frequency axis
sampling_rate = 1  # Adjust according to your actual sampling rate
freq = np.fft.fftfreq(fft_data.shape[1], d=sampling_rate)

# Prepare features and labels for machine learning
X = fft_magnitude
y = labels

# Split the data into training and testing setsa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
     
    # 'KNN': KNeighborsClassifier(),
    'Random Forest' : RandomForestClassifier()
    # 'Logistic Regression': LogisticRegression(max_iter=1000),
    # 'MultinomialNB': MultinomialNB(),
    # 'GaussianNB': GaussianNB(),
    # 'BernoulliNB': BernoulliNB(),
    # 'DecisionTree': DecisionTreeClassifier(),
    # 'Neural Network': MLPClassifier(),
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
    print(classification_report(y_test, y_pred,zero_division=0))


# Plotting the FFT magnitude of each row with different colors
colors = {'normal.': 'blue', 'smurf.': 'black', 'neptune.': 'red', 'back': 'brown', 'pod': 'pink', 'teardrop': 'yellow', 'buffer_overflow': 'green', 'warezclient': 'lightblue', 'ipsweep': 'darkred', 'portsweep': 'aqua', 'satan': 'blueviolet'}

plt.figure(figsize=(14, 6))
for i in range(len(df)):
    if labels[i] in colors:
        plt.plot(freq[:len(fft_magnitude[i]) // 2], fft_magnitude[i][:len(fft_magnitude[i]) // 2], color=colors[labels[i]], linewidth=2.5)

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
plt.title("Frequency Distribution")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.yscale('log')
plt.tight_layout()
plt.show()