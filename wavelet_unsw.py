import numpy as np
import pywt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Function to preprocess data and apply wavelet transform with padding
def preprocess_data(filepath, encoder=None, fit_encoder=False, pad_length=3000):
    df = pd.read_csv(filepath)
    
    # Drop unnecessary columns (e.g., index or non-feature columns)
    df.drop(columns=['id'], inplace=True, errors='ignore')  # Adjust column name as needed

    # Handle categorical columns with OneHotEncoding
    if fit_encoder:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_encoded = encoder.fit_transform(df[categorical_cols])
        df_encoded = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        df_combined = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)
    else:
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_encoded = encoder.transform(df[categorical_cols])
        df_encoded = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        df_combined = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)

    # Ensure all data used for wavelet transform is numeric
    numeric_df = df_combined.select_dtypes(include=[np.number])

    # Convert numeric data to numpy array
    data_numeric = numeric_df.values

    # Apply Wavelet Transform and extract features
    wavelet_coeffs, additional_features = [], []
    for row in data_numeric:
        try:
            coeffs = pywt.wavedec(row, 'haar')
            coeffs_flat = np.hstack(coeffs)
            
            # Pad coefficients to ensure consistent length
            coeffs_flat = np.pad(coeffs_flat, (0, max(0, pad_length - len(coeffs_flat))), 'constant')[:pad_length]

            # Calculate statistical features
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

            # Store features
            wavelet_coeffs.append(coeffs_flat)
            additional_features.append([
                np.mean(percentage_dev), entropy, skewness_coeffs, kurtosis_coeffs,
                mean_coeffs, std_dev_coeffs, var_coeffs, max_coeffs, min_coeffs,
                range_coeffs, mad_coeffs
            ])
        except Exception as e:
            print(f"Error processing row {row}: {e}")

    # Combine wavelet coefficients and additional features
    wavelet_coeffs = np.array(wavelet_coeffs)
    additional_features = np.array(additional_features)
    final_data = np.hstack([wavelet_coeffs, additional_features])

    # Extract labels (assuming 'label' column is the target)
    labels = df['label'].astype('category').cat.codes

    return final_data, labels, encoder

# Load and preprocess training and testing data separately
X_train, y_train, encoder = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\unsw_train.txt", fit_encoder=True)

X_test, y_test, _ = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\unsw_test.txt", encoder=encoder, fit_encoder=False)

# Normalize/Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Feature Selection with SelectKBest
kbest = SelectKBest(score_func=f_classif, k=50)
X_train_kbest = kbest.fit_transform(X_train_pca, y_train)
X_test_kbest = kbest.transform(X_test_pca)

# Continue with training classifiers...


# Initialize classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'DecisionTree': DecisionTreeClassifier(),
    'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10),
    'Boosting': AdaBoostClassifier(estimator=RandomForestClassifier(), n_estimators=1000),
    'KNN': KNeighborsClassifier(n_neighbors=125),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
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
    print(f"Test Accuracy of {name}: {accuracy_test:.4f}")
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
