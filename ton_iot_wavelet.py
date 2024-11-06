import numpy as np
import pywt
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

def preprocess_data(filepath):

    df = pd.read_csv(filepath)
    
    # Identify features and labels
    features = df.drop(columns=['Label', 'Attack'])  # Drop the label and attack columns
    labels = df['Label'].values  # The 'Label' column contains the target labels

    # Standardize numerical features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply Wavelet Transform and extract features
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

    for row in features_scaled:
        coeffs = pywt.wavedec(row, 'db4', level=4)  # Use 'db4' wavelet
        coeffs_flat = np.hstack(coeffs)

        # Compute additional statistical features
        mean_coeffs = np.mean(coeffs_flat)
        std_dev_coeffs = np.std(coeffs_flat)
        var_coeffs = np.var(coeffs_flat)
        max_coeffs = np.max(coeffs_flat)
        min_coeffs = np.min(coeffs_flat)
        range_coeffs = max_coeffs - min_coeffs
        mad_coeffs = np.mean(np.abs(coeffs_flat - mean_coeffs))

        percentage_dev = np.abs(coeffs_flat - mean_coeffs) / mean_coeffs * 100

        # Compute entropy
        entropy = stats.entropy(np.abs(coeffs_flat) + 1e-9)  # Add small value to avoid log(0)

        # Compute skewness and kurtosis
        skewness_coeffs = stats.skew(coeffs_flat)
        kurtosis_coeffs = stats.kurtosis(coeffs_flat)

        # Store features
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

    additional_features = np.column_stack([ 
        percentage_deviation, entropy_values, skewness_values, kurtosis_values,
        mean_values, std_dev_values, var_values, max_values, min_values,
        range_values, mad_values
    ])

    final_data = np.hstack([wavelet_coeffs, additional_features])

    return final_data, labels
# Load and preprocess data
X,y = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\NF-ToN-IoT.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize/Standardize the data
scaler = StandardScaler()
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
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'DecisionTree': DecisionTreeClassifier(),
    'Boosting': AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=10),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}

# Prepare to store evaluation metrics
evaluation_metrics = {'Classifier': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train_kbest, y_train)

    # Evaluate on test data
    y_test_pred = clf.predict(X_test_kbest)

    # Metrics calculation
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # Confusion Matrix
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Plot confusion matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title(f"Test Confusion Matrix - {name}")
    plt.show()

    # Print classification report
    print(f"Test Accuracy of {name}: {accuracy_test:.7f}")
    print(classification_report(y_test, y_test_pred, zero_division=0))
