import numpy as np
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import catboost
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
import time
import scipy.stats as stats
import xgboost as xgb
from codecarbon import EmissionsTracker
from codecarbon import EmissionsTracker
import warnings
from sklearn.neighbors import KNeighborsClassifier


# Suppress User and Runtime warnings if necessary
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

    # Extract labels (Assuming last but one column contains the attack class)
    labels = df[df.columns[-2]]

    # Convert labels to binary (1 if attack, 0 if normal)
    labels = np.where(labels != 'normal', 1, 0)  # Change 'normal.' based on your dataset specifics

    # Convert numeric data to numpy array
    data_numeric = df_combined.values

    # Apply Wavelet Transform and extract features
    wavelet_coeffs = []
    additional_features = []

    for row in data_numeric:
        coeffs = pywt.wavedec(row, 'haar', level=4)  # Use 'db4' wavelet
        approx_coeffs = coeffs[0]
        detail_coeffs = coeffs[1:]

        # Combine statistical features on each subband (approximation + details)
        subband_stats = []
        for subband in [approx_coeffs] + detail_coeffs:
            mean_subband = np.mean(subband)
            std_dev_subband = np.std(subband)
            var_subband = np.var(subband)
            max_subband = np.max(subband)
            min_subband = np.min(subband)
            range_subband = max_subband - min_subband
            mad_subband = np.mean(np.abs(subband - mean_subband))
            skewness_subband = stats.skew(subband)
            kurtosis_subband = stats.kurtosis(subband)
            subband_stats.extend([mean_subband, std_dev_subband, var_subband, max_subband, min_subband,
                                  range_subband, mad_subband, skewness_subband, kurtosis_subband])

        wavelet_coeffs.append(np.hstack(coeffs))
        additional_features.append(subband_stats)

    wavelet_coeffs = np.array(wavelet_coeffs)
    additional_features = np.array(additional_features)

    final_data = np.hstack([wavelet_coeffs, additional_features])

    return final_data, labels, encoder

# # Load and preprocess training and testing data
# X_train_data, y_train_data, encoder = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+.txt", fit_encoder=True)
# X_test_data, y_test_data, _ = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTest+.txt", encoder=encoder)

# # Combine train and test datasets
# X_combined = np.vstack([X_train_data, X_test_data])
# y_combined = np.hstack([y_train_data, y_test_data])

# # Now split the combined data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)


# Load and preprocess training data from KDDTrain+
X_train, y_train, encoder = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+.txt", fit_encoder=True)

# Load and preprocess test data from KDDTest+
X_test, y_test, _ = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTest+.txt", encoder=encoder)

# Normalize/Standardize the data
scaler =  StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Feature Selection with SelectKBest
kbest = SelectKBest(score_func=f_classif, k=25)
X_train_kbest = kbest.fit_transform(X_train_pca, y_train)
X_test_kbest = kbest.transform(X_test_pca)

# Initialize classifiers with default parameters
classifiers = {
    'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'DecisionTree': DecisionTreeClassifier(),
    'Boosting': AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=10),
    'CatBoost': catboost.CatBoostClassifier(learning_rate=0.1, depth=6, iterations=100, verbose=0),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train_kbest, y_train)

    # Evaluate on test data
    y_test_pred = clf.predict(X_test_kbest)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Plot confusion matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title(f"Test Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Print accuracy and classification report
    print(f"Test Accuracy of {name}: {accuracy_test:.7f}")
    print(classification_report(y_test, y_test_pred, zero_division=0))
