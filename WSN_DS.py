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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
import time
import scipy.stats as stats
import xgboost as xgb
from codecarbon import EmissionsTracker

# Function to preprocess WSN data and apply wavelet transform
def preprocess_data_wsn(filepath, encoder=None, fit_encoder=False):
    df = pd.read_csv(filepath)

    # Convert categorical column to numeric codes
    if fit_encoder:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_column = df[df.columns[-1]]
        categorical_encoded = encoder.fit_transform(categorical_column.values.reshape(-1, 1))
        df_encoded = pd.DataFrame(categorical_encoded, index=df.index, columns=encoder.get_feature_names_out())
        df_combined = pd.concat([df.drop(columns=[df.columns[-1]]), df_encoded], axis=1)
    else:
        df_combined = df.drop(columns=[df.columns[-1]])
    
    # Ensure all data used for wavelet transform is numeric
    numeric_df = df_combined.select_dtypes(include=[np.number])

    # Convert numeric data to numpy array
    data_numeric = numeric_df.values

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
        try:
            # Apply wavelet transform only if data is valid
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
        except Exception as e:
            print(f"Error processing row {row}: {e}")

    # Convert features to numpy array
    wavelet_coeffs = np.array(wavelet_coeffs)

    additional_features = np.column_stack([ 
        percentage_deviation, entropy_values, skewness_values, kurtosis_values,
        mean_values, std_dev_values, var_values, max_values, min_values,
        range_values, mad_values
    ])

    final_data = np.hstack([wavelet_coeffs, additional_features])

    # Extract labels
    labels = df[df.columns[-1]].astype('category').cat.codes

    return final_data, labels

# Load and preprocess data
X, y = preprocess_data_wsn("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\WSN-DS.csv")

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

# Initialize classifiers
classifiers = {
    # Adjusted MLP classifier with smaller hidden layers and fewer iterations
    # Adjusted MLP classifier with smaller hidden layers and fewer iterations
    'MLP': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300, learning_rate_init=0.001, random_state=42)

    # 'MLP': MLPClassifier(hidden_layer_sizes=(20,), max_iter=200, learning_rate_init=0.005, random_state=42)

    # 'RandomForest': RandomForestClassifier(n_estimators=100),
    # 'DecisionTree': DecisionTreeClassifier(),
    # 'Boosting': AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=10),
    # 'CatBoost': catboost.CatBoostClassifier(learning_rate=0.1, depth=6, iterations=100, verbose=0),
}

# Store metrics for plotting
results = {
    'Classifier': [],
    'Accuracy': [],
    'F1 Score': [],
    'Recall': []
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")

    clf.fit(X_train_kbest, y_train)

    # Evaluate on test data
    y_test_pred = clf.predict(X_test_kbest)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred, average='weighted')
    recall_test = recall_score(y_test, y_test_pred, average='weighted')

    # # Append metrics to results
    # results['Classifier'].append(name)
    # results['Accuracy'].append(accuracy_test)
    # results['F1 Score'].append(f1_test)
    # results['Recall'].append(recall_test)

    # Plot confusion matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title(f"Test Confusion Matrix - {name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Print accuracy and classification report
    print(f"Test Accuracy of {name}: {accuracy_test:.7f}")
    print(classification_report(y_test, y_test_pred, zero_division=0))

# Convert results to DataFrame for easier plotting
# results_df = pd.DataFrame(results)

# # Plotting metrics
# plt.figure(figsize=(10, 6))

# # Accuracy plot
# plt.subplot(3, 1, 1)
# sns.barplot(data=results_df, x='Accuracy', y='Classifier', palette='viridis')
# plt.title('Accuracy of Classifiers')

# # F1 Score plot
# plt.subplot(3, 1, 2)
# sns.barplot(data=results_df, x='F1 Score', y='Classifier', palette='viridis')
# plt.title('F1 Score of Classifiers')

# # Recall plot
# plt.subplot(3, 1, 3)
# sns.barplot(data=results_df, x='Recall', y='Classifier', palette='viridis')
# plt.title('Recall of Classifiers')

# plt.tight_layout()
# plt.show()
