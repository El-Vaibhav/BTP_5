import pandas as pd
import numpy as np
import pywt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
import time
import scipy.stats as stats
from codecarbon import EmissionsTracker


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
    l = [i for i in range(1, 23)]
    l.append(df.columns[-2])

    df_combined = pd.concat([df.drop(columns=l), df_encoded], axis=1)

    # Extract labels
    labels = df[df.columns[-2]]

    # Convert numeric data to numpy array
    data_numeric = df_combined.values

    # Apply Wavelet Transform
    wavelet_coeffs = []
    percentage_deviation = []
    entropy_values = []
    skewness_values = []
    kurtosis_values = []

    for row in data_numeric:
        coeffs = pywt.wavedec(row, 'db4', level=4)  # Use 'db4' wavelet
        coeffs_flat = np.hstack(coeffs)

        # computing statistical features
        skewness_coeffs = stats.skew(coeffs_flat)
        kurtosis_coeffs = stats.kurtosis(coeffs_flat)

        # Compute percentage deviation
        mean_coeffs = np.mean(coeffs_flat)
        percent_dev = np.abs(coeffs_flat - mean_coeffs) / mean_coeffs * 100
        percentage_deviation.append(np.mean(percent_dev))  # Store mean percentage deviation

        # Compute entropy
        entropy = stats.entropy(np.abs(coeffs_flat))
        entropy_values.append(entropy)

        # compute statistical values
        skewness_values.append(skewness_coeffs)
        kurtosis_values.append(kurtosis_coeffs)
        wavelet_coeffs.append(coeffs_flat)

    # Convert coefficients to numpy array
    wavelet_coeffs = np.array(wavelet_coeffs)

    additional_features = np.column_stack([percentage_deviation, entropy_values, skewness_values, kurtosis_values])

    final_data = np.hstack([wavelet_coeffs, additional_features])

    return final_data, labels, encoder


# Load and preprocess data
X, y, encoder = preprocess_data("C:\\Users\\tabis\\OneDrive\\Desktop\\BTP_5\\KDDTrain+.txt", fit_encoder=True)

# Normalize/Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Feature Selection with SelectKBest
kbest = SelectKBest(score_func=f_classif, k=25)
X_train_kbest = kbest.fit_transform(X_train_pca, y_train)
X_test_kbest = kbest.transform(X_test_pca)

# Initialize classifiers with default parameters
classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")

    # Start CodeCarbon tracker
    tracker = EmissionsTracker()
    tracker.start()

    clf.fit(X_train_kbest, y_train)



    # Evaluate on test data
    y_test_pred = clf.predict(X_test_kbest)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Stop CodeCarbon tracker and get energy consumption
    emissions = tracker.stop()

    # Print energy consumption
    print(f"Energy consumed for {name}: {emissions:.6f} kWh")

    # Plot confusion matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title(f"Test Confusion Matrix - {name}")
    plt.show()

    # Print accuracy and classification report
    print(f"Test Accuracy of {name}: {accuracy_test:.7f}")
    print(classification_report(y_test, y_test_pred, zero_division=0))

