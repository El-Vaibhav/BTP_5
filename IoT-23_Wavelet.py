import numpy as np
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Function to apply Wavelet Transform and extract statistical features
def apply_wavelet_transform(data_numeric):
    wavelet_coeffs = []
    percentage_deviation = []
    entropy_values = []
    skewness_values = []
    kurtosis_values = []
    mean_values = []
    std_dev_values = []
    var_values = []
    max_values = []
    min_values = []
    range_values = []
    mad_values = []  # Mean Absolute Deviation

    for row in data_numeric:
        coeffs = pywt.wavedec(row, 'db4', level=4)  # Use 'db4' wavelet
        coeffs_flat = np.hstack(coeffs)

        # Compute statistical features
        mean_coeffs = np.mean(coeffs_flat)
        std_dev_coeffs = np.std(coeffs_flat)
        var_coeffs = np.var(coeffs_flat)
        max_coeffs = np.max(coeffs_flat)
        min_coeffs = np.min(coeffs_flat)
        range_coeffs = max_coeffs - min_coeffs
        mad_coeffs = np.mean(np.abs(coeffs_flat - mean_coeffs))

        percentage_dev = np.abs(coeffs_flat - mean_coeffs) / mean_coeffs * 100
        entropy = stats.entropy(np.abs(coeffs_flat))
        skewness_coeffs = stats.skew(coeffs_flat)
        kurtosis_coeffs = stats.kurtosis(coeffs_flat)

        wavelet_coeffs.append(coeffs_flat)
        percentage_deviation.append(np.mean(percentage_dev))
        entropy_values.append(entropy)
        skewness_values.append(skewness_coeffs)
        kurtosis_values.append(kurtosis_coeffs)
        mean_values.append(mean_coeffs)
        std_dev_values.append(std_dev_coeffs)
        var_values.append(var_coeffs)
        max_values.append(max_coeffs)
        min_values.append(min_coeffs)
        range_values.append(range_coeffs)
        mad_values.append(mad_coeffs)

    # Convert features to numpy array
    wavelet_coeffs = np.array(wavelet_coeffs)

    additional_features = np.column_stack([percentage_deviation, entropy_values, skewness_values, kurtosis_values,
                                           mean_values, std_dev_values, var_values, max_values, min_values,
                                           range_values, mad_values])

    final_data = np.hstack([wavelet_coeffs, additional_features])
    return final_data

# Function to load and preprocess the dataset
def preprocess_iot_data(filepath):

    df = pd.read_csv(filepath)  # Load IoT-23 dataset

    # Extract labels (modify the label column name)
    labels = df['label']  # Replace 'label' with the actual column name for labels

    # Drop the label column and any irrelevant categorical columns if necessary
    df_numeric = df.drop(columns=['label','id.orig_h'])  # Modify column names as needed

    # Convert numeric data to numpy array
    data_numeric = df_numeric.values

    # Apply Wavelet Transform and extract features
    final_data = apply_wavelet_transform(data_numeric)

    return final_data, labels

# Load and preprocess training data
X_train, y_train = preprocess_iot_data("C:\\Users\\tabis\\OneDrive\\Desktop\\Datasets\\iot23_combined.csv")

# Load and preprocess test data
X_test, y_test = preprocess_iot_data("C:\\Users\\tabis\\OneDrive\\Desktop\\Datasets\\iot23_combined.csv")

# The rest of the pipeline (scaling, PCA, feature selection, and classification)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

kbest = SelectKBest(score_func=f_classif, k=25)
X_train_kbest = kbest.fit_transform(X_train_pca, y_train)
X_test_kbest = kbest.transform(X_test_pca)

classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=100),
}

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train_kbest, y_train)
    y_test_pred = clf.predict(X_test_kbest)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title(f"Test Confusion Matrix - {name}")
    plt.show()

    print(f"Test Accuracy of {name}: {accuracy_test:.7f}")
    print(classification_report(y_test, y_test_pred, zero_division=0))
