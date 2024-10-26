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

    df_combined = pd.concat([df.drop(columns=[1,2,3,df.columns[-2]]), df_encoded], axis=1)

    # print(df_combined.describe())

    # Extract labels
    labels = df[df.columns[-2]]

    # Convert numeric data to numpy array
    data_numeric = df_combined.values

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

    for row in data_numeric:
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
        entropy = stats.entropy(np.abs(coeffs_flat))

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

    return final_data, labels, encoder

# Load and preprocess data
# X, y, encoder = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+.txt", fit_encoder=True)

# # Load and preprocess training data from KDDTrain+
# X_train, y_train, encoder = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+.txt", fit_encoder=True)

# # Load and preprocess test data from KDDTest+
# X_test, y_test, _ = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTest+.txt", encoder=encoder)

#  Load and preprocess both training and testing data
X_train_data, y_train_data, encoder = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+.txt", fit_encoder=True)
X_test_data, y_test_data, _ = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTest+.txt", encoder=encoder)

# Combine train and test datasets
X_combined = np.vstack([X_train_data, X_test_data])
y_combined = np.hstack([y_train_data, y_test_data])

# Now split the combined data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)

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
    # 'SVM': SVC(),
    # 'KNN': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(),
    # 'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10),
    'Boosting': AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=10),
    'CatBoost': catboost.CatBoostClassifier(learning_rate=0.1, depth=6, iterations=100, verbose=0),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}


# # Train and evaluate each classifier
# for name, clf in classifiers.items():
#     print(f"Training {name}...")

#     # Start CodeCarbon tracker
#     # tracker = EmissionsTracker()
#     # tracker.start()

#     clf.fit(X_train_kbest, y_train)


#     # Evaluate on test data
#     y_test_pred = clf.predict(X_test_kbest)
#     accuracy_test = accuracy_score(y_test, y_test_pred)
#     cm_test = confusion_matrix(y_test, y_test_pred)

#     # Stop CodeCarbon tracker and get energy consumption
#     # emissions = tracker.stop()

#     # # Print energy consumption
#     # print(f"Energy consumed for {name}: {emissions:.6f} kWh")

#     # Plot confusion matrix
#     sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
#                 yticklabels=np.unique(y_test))
#     plt.title(f"Test Confusion Matrix - {name}")
#     plt.show()

#     # Print accuracy and classification report
#     print(f"Test Accuracy of {name}: {accuracy_test:.7f}")
#     print(classification_report(y_test, y_test_pred, zero_division=0))

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

# Dictionary to store evaluation metrics for each classifier
evaluation_metrics = {
    'Classifier': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")

    clf.fit(X_train_kbest, y_train)

    # Evaluate on test data
    y_test_pred = clf.predict(X_test_kbest)

    # Accuracy
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # Precision, Recall, F1-Score (weighted to handle imbalance)
    precision_test = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    recall_test = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1_test = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    # Store metrics
    evaluation_metrics['Classifier'].append(name)
    evaluation_metrics['Accuracy'].append(accuracy_test)
    evaluation_metrics['Precision'].append(precision_test)
    evaluation_metrics['Recall'].append(recall_test)
    evaluation_metrics['F1-Score'].append(f1_test)

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

# Preprocess and load KDDTrain+_20Percent dataset
X_train_20p, y_train_20p, encoder_20p = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+_20Percent.txt", fit_encoder=True)
X_test_data, y_test_data, _ = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTest+.txt", encoder=encoder_20p)

# Combine train and test datasets for KDDTrain+_20Percent
X_test_data_truncated = X_test_data[:, :158] 
X_combined_20p = np.vstack([X_train_20p, X_test_data_truncated])
y_combined_20p = np.hstack([y_train_20p, y_test_data])

# Now split the combined data into train and test sets
X_train_20p, X_test_20p, y_train_20p, y_test_20p = train_test_split(X_combined_20p, y_combined_20p, test_size=0.3, random_state=42)

# Normalize/Standardize the data
X_train_scaled_20p = scaler.fit_transform(X_train_20p)
X_test_scaled_20p = scaler.transform(X_test_20p)

# Apply PCA
X_train_pca_20p = pca.fit_transform(X_train_scaled_20p)
X_test_pca_20p = pca.transform(X_test_scaled_20p)

# Apply SelectKBest
X_train_kbest_20p = kbest.fit_transform(X_train_pca_20p, y_train_20p)
X_test_kbest_20p = kbest.transform(X_test_pca_20p)

# Dictionary to store evaluation metrics for both datasets
evaluation_metrics_20p = {
    'Classifier': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

# Train and evaluate each classifier on KDDTrain+_20Percent
for name, clf in classifiers.items():
    print(f"Training {name} on KDDTrain+_20Percent...")

    clf.fit(X_train_kbest_20p, y_train_20p)

    # Evaluate on test data
    y_test_pred_20p = clf.predict(X_test_kbest_20p)

    # Accuracy, Precision, Recall, F1-Score
    accuracy_test_20p = accuracy_score(y_test_20p, y_test_pred_20p)
    precision_test_20p = precision_score(y_test_20p, y_test_pred_20p, average='weighted', zero_division=0)
    recall_test_20p = recall_score(y_test_20p, y_test_pred_20p, average='weighted', zero_division=0)
    f1_test_20p = f1_score(y_test_20p, y_test_pred_20p, average='weighted', zero_division=0)

    # Store metrics for KDDTrain+_20Percent
    evaluation_metrics_20p['Classifier'].append(name)
    evaluation_metrics_20p['Accuracy'].append(accuracy_test_20p)
    evaluation_metrics_20p['Precision'].append(precision_test_20p)
    evaluation_metrics_20p['Recall'].append(recall_test_20p)
    evaluation_metrics_20p['F1-Score'].append(f1_test_20p)

    cm_test = confusion_matrix(y_test_20p, y_test_pred_20p)
    
    # Plot confusion matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test_20p),
                yticklabels=np.unique(y_test_20p))
    plt.title(f"Test Confusion Matrix - {name}")
    plt.show()

    # Print classification report
    print(f"Test Accuracy of {name}: {accuracy_test_20p:.7f}")
    print(classification_report(y_test_20p, y_test_pred_20p, zero_division=0))

# Plot metrics comparison between KDDTrain+ and KDDTrain+_20Percent
def plot_comparison(metric_name, values_full, values_20p, color1, color2):
    bar_width = 0.35  # Set thinner bar width
    index = np.arange(len(evaluation_metrics['Classifier']))

    plt.figure(figsize=(12, 7))
    plt.bar(index, values_full, bar_width, label='KDDTrain+', color=color1)
    plt.bar(index + bar_width, values_20p, bar_width, label='KDDTrain+_20Percent', color=color2)

    plt.xlabel('Classifiers')
    plt.ylabel(metric_name)
    plt.title(f'Comparison of {metric_name} between KDDTrain+ and KDDTrain+_20Percent')
    plt.xticks(index + bar_width / 2, evaluation_metrics['Classifier'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot bar graphs for each metric comparison
plot_comparison('Accuracy', evaluation_metrics['Accuracy'], evaluation_metrics_20p['Accuracy'], 'blue', 'orange')
plot_comparison('Precision', evaluation_metrics['Precision'], evaluation_metrics_20p['Precision'], 'green', 'pink')
plot_comparison('Recall', evaluation_metrics['Recall'], evaluation_metrics_20p['Recall'], 'red', 'cyan')
plot_comparison('F1-Score', evaluation_metrics['F1-Score'], evaluation_metrics_20p['F1-Score'], 'goldenrod', 'purple')


