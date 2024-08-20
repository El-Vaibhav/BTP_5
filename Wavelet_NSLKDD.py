import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Function to preprocess data and apply wavelet transform
def preprocess_data(filepath, encoder=None, fit_encoder=False):
    df = pd.read_csv(filepath, header=None)
    
    # Extract categorical columns (columns 1, 2, 3) and apply one-hot encoding
    categorical_columns = df[[1, 2, 3]]
    if fit_encoder:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_encoded = encoder.fit_transform(categorical_columns)
    else:
        categorical_encoded = encoder.transform(categorical_columns)
    
    # Convert encoded data to DataFrame
    df_encoded = pd.DataFrame(categorical_encoded, index=df.index)
    
    # Combine one-hot encoded columns with the rest of the data (numeric columns)
    df_combined = pd.concat([df.drop(columns=[1, 2, 3, df.columns[-2]]), df_encoded], axis=1)
    
    # Extract labels
    labels = df[df.columns[-2]]
    
    # Convert numeric data to numpy array
    data_numeric = df_combined.values
    
    # Apply Wavelet Transform
    wavelet_coeffs = []
    for row in data_numeric:
        coeffs = pywt.wavedec(row, 'db4', level=4)  # Use 'db4' wavelet
        coeffs_flat = np.hstack(coeffs) 
        wavelet_coeffs.append(coeffs_flat)
    
    # Convert coefficients to numpy array
    wavelet_coeffs = np.array(wavelet_coeffs)
    
    return wavelet_coeffs, labels, encoder

# Load and preprocess data
X, y, encoder = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+.txt", fit_encoder=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize classifiers with default parameters
classifiers = {
    # 'KNN': KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    # 'DecisionTree': DecisionTreeClassifier(),
    # 'SVM': SVC()
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    
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

# Plotting the Wavelet coefficients of each row with different colors
# colors = {'normal.': 'blue', 'smurf.': 'black', 'neptune.': 'red', 'back': 'brown', 'pod': 'pink', 'teardrop': 'yellow', 'buffer_overflow': 'green', 'warezclient': 'lightblue', 'ipsweep': 'darkred', 'portsweep': 'aqua', 'satan': 'blueviolet'}

# plt.figure(figsize=(14, 6))
# for i in range(len(X_train)):
#     if y_train.iloc[i] in colors:
#         plt.plot(np.arange(X_train.shape[1]), X_train[i], color=colors[y_train.iloc[i]], linewidth=2.5)

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
# plt.title("Wavelet Coefficients")
# plt.xlabel("Coefficient Index")
# plt.ylabel("Magnitude")
# plt.yscale('log')
# plt.tight_layout()
# plt.show()
