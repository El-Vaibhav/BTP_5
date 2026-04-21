import numpy as np
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier,StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import xgboost as xgb
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
X, y, encoder = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+_20Percent.txt", fit_encoder=True)

# Normalize/Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Load and preprocess training data
X_train, y_train, encoder = preprocess_data("C:\\Users\\tabis\\OneDrive\\Desktop\\Datasets\\KDDTrain+.txt", fit_encoder=True)

# Load and preprocess test data using the same encoder
X_test, y_test, _ = preprocess_data("C:\\Users\\tabis\\OneDrive\\Desktop\\Datasets\\KDDTest+.txt", encoder=encoder)

# Normalize the data using MinMaxScaler (scaling each feature to a range [0, 1])
scaler = MinMaxScaler()
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


# The base learners provide predictions on the data, and the final estimator uses those predictions to make a final decision.


# Define base learners (RandomForest, DecisionTree, MLP)
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('dt', DecisionTreeClassifier()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)),
]

# Create the final estimator (XGBoost)
final_estimator = xgb.XGBClassifier()

# Create StackingClassifier
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=final_estimator
)

# Train the hybrid model
stacking_clf.fit(X_train_kbest, y_train)

# Evaluate on test data
y_test_pred = stacking_clf.predict(X_test_kbest)
accuracy_test = accuracy_score(y_test, y_test_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title(f"Test Confusion Matrix - Stacking Classifier")
plt.show()

# Print accuracy and classification report
print(f"Test Accuracy of Stacking Classifier: {accuracy_test:.7f}")
print(classification_report(y_test, y_test_pred, zero_division=0))

# Initialize classifiers with default parameters
classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=100),
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
    plt.show()

    # Print accuracy and classification report
    print(f"Test Accuracy of {name}: {accuracy_test:.7f}")
    print(classification_report(y_test, y_test_pred, zero_division=0))

