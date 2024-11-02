import numpy as np
import pandas as pd
import pywt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import catboost
import scipy.stats as stats

def preprocess_data_iot(filepath):
    df = pd.read_csv(filepath)

    # Separate features and labels
    features = df.drop(columns=['label'])
    labels = df['label']

    # No categorical encoding as there are no categorical features
    features_combined = features.values  # Use raw feature values directly

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

    for row in features_combined:
        try:
            # Apply wavelet transform
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
            entropy = stats.entropy(np.abs(coeffs_flat) + 1e-10)  # Adding a small value to avoid division by zero

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
    labels = df['label'].astype('category').cat.codes

    return final_data, labels

# Load and preprocess the data
X, y = preprocess_data_iot("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\iot23_combined.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'DecisionTree': DecisionTreeClassifier(),
    'Boosting': AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=10),
    'CatBoost': catboost.CatBoostClassifier(learning_rate=0.1, depth=6, iterations=100, verbose=0),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
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

    # Append metrics to results
    results['Classifier'].append(name)
    results['Accuracy'].append(accuracy_test)
    results['F1 Score'].append(f1_test)
    results['Recall'].append(recall_test)

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
results_df = pd.DataFrame(results)

# Plotting metrics
plt.figure(figsize=(10, 6))

# Accuracy plot
plt.subplot(3, 1, 1)
sns.barplot(data=results_df, x='Accuracy', y='Classifier', palette='viridis')
plt.title('Accuracy of Classifiers')

# F1 Score plot
plt.subplot(3, 1, 2)
sns.barplot(data=results_df, x='F1 Score', y='Classifier', palette='viridis')
plt.title('F1 Score of Classifiers')

# Recall plot
plt.subplot(3, 1, 3)
sns.barplot(data=results_df, x='Recall', y='Classifier', palette='viridis')
plt.title('Recall of Classifiers')

plt.tight_layout()
plt.show()
