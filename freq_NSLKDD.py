import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Read the dataset
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\KDDTrain+.txt", header=None)

# Display the first few rows of the dataframe
print(df.head())

# Drop non-numeric columns and extract numeric data
df_numeric = df.drop(columns=[1, 2, 3, df.columns[-2]])

# Convert remaining columns to float
df_numeric = df_numeric.astype(float)

# Extract labels
labels = df[df.columns[-2]]

# Convert numeric data to numpy array
data_numeric = df_numeric.to_numpy()

# Ensure all rows have the same length by padding shorter rows
max_length = max(len(row) for row in data_numeric)
padded_data_numeric = np.array([np.pad(row, (0, max_length - len(row))) for row in data_numeric])

# Apply Hamming window to reduce spectral leakage
window = windows.hamming(max_length)
data_numeric_windowed = padded_data_numeric * window

# # Apply FFT along each row (axis=1)
# fft_data = np.fft.fft(data_numeric_windowed, axis=1)

# # Take the magnitude of the FFT results
# fft_magnitude = np.abs(fft_data)

# # Define frequency axis
# sampling_rate = 1  # Adjust according to your actual sampling rate
# freq = np.fft.fftfreq(max_length, d=sampling_rate)

X = data_numeric_windowed
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
     
    # 'KNN': KNeighborsClassifier(),
    # 'Logistic Regression': LogisticRegression(max_iter=1000),
    # 'MultinomialNB': MultinomialNB(),
    # 'GaussianNB': GaussianNB(),
    # 'BernoulliNB': BernoulliNB(),
    # 'DecisionTree': DecisionTreeClassifier(),
    # 'Neural Network': MLPClassifier(),
    "Support Vector Machine": SVC(kernel='linear', C=1.0)
    # 'Random Forest' : RandomForestClassifier(),
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

# Define colors for different labels
# colors = {
#     'normal': 'blue',
#     'smurf': 'black',
#     'neptune': 'red',
#     'ipsweep': 'green',
#     'portsweep': 'orange',
#     'warezclient': 'purple',
#     'satan': 'brown',
#     'teardrop': 'pink'
# }

# # Plotting the FFT magnitude of each row with different colors
# plt.figure(figsize=(14, 6))
# for i in range(len(df)):
#     if labels[i] in colors:
#         plt.plot(freq[:len(fft_magnitude[i])//2], fft_magnitude[i][:len(fft_magnitude[i])//2], color=colors[labels[i]], linewidth=2.5)

# # Create custom legend
# legend_entries = [
#     plt.Line2D([0], [0], color='blue', linewidth=2.5, label='normal'),
#     plt.Line2D([0], [0], color='black', linewidth=2.5, label='smurf (DOS attack)'),
#     plt.Line2D([0], [0], color='red', linewidth=2.5, label='neptune (DOS attack)'),
#     plt.Line2D([0], [0], color='green', linewidth=2.5, label='ipsweep (reconnaissance attack)'),
#     plt.Line2D([0], [0], color='orange', linewidth=2.5, label='portsweep (reconnaissance attack)'),
#     plt.Line2D([0], [0], color='purple', linewidth=2.5, label='warezclient (reconnaissance attack)'),
#     plt.Line2D([0], [0], color='brown', linewidth=2.5, label='satan (reconnaissance attack)'),
#     plt.Line2D([0], [0], color='pink', linewidth=2.5, label='teardrop (DOS attack)')
# ]

# plt.legend(handles=legend_entries, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)

# # Adjust plot layout
# plt.tight_layout()

# plt.title("Frequency Distribution")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.yscale('log')  # Use log scale if magnitude varies widely
# plt.show()

# colors = {'normal.': 'blue', 'smurf.': 'black', 'neptune.': 'red', 'back': 'brown', 'pod': 'pink', 'teardrop': 'yellow', 'buffer_overflow': 'green', 'warezclient': 'lightblue', 'ipsweep': 'darkred', 'portsweep': 'aqua', 'satan': 'blueviolet'}

# plt.figure(figsize=(14, 6))
# for i in range(len(df)):
#     if labels[i] in colors:
#         plt.plot(np.arange(len(fft_magnitude[i])), fft_magnitude[i], color=colors[labels[i]], linewidth=2.5)

# legend_entries = [
#     plt.Line2D([0], [0], color='blue', label='normal', linewidth=2.5),
#     plt.Line2D([0], [0], color='black', label='smurf (DOS attack)', linewidth=2.5),
#     plt.Line2D([0], [0], color='red', label='neptune (DOS attack)', linewidth=2.5),
#     plt.Line2D([0], [0], color='brown', label='back (DOS attack)', linewidth=2.5),
#     plt.Line2D([0], [0], color='pink', label='pod (DOS attack)', linewidth=2.5),
#     plt.Line2D([0], [0], color='yellow', label='teardrop (DOS attack)', linewidth=2.5),
#     plt.Line2D([0], [0], color='green', label='buffer_overflow (U2R attack)', linewidth=2.5),
#     plt.Line2D([0], [0], color='lightblue', label='warezclient (R2L attack)', linewidth=2.5),
#     plt.Line2D([0], [0], color='darkred', label='ipsweep (Probe attack)', linewidth=2.5),
#     plt.Line2D([0], [0], color='aqua', label='portsweep (Probe attack)', linewidth=2.5),
#     plt.Line2D([0], [0], color='blueviolet', label='satan (Probe attack)', linewidth=2.5),
# ]

# plt.legend(handles=legend_entries, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
# plt.title("FFT Magnitude")
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.yscale('log')
# plt.tight_layout()
# plt.show()
