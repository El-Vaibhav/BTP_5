import streamlit as st
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy.stats as stats
from codecarbon import EmissionsTracker

# Suppress User and Runtime warnings if necessary
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Streamlit app title
st.title("Wavelet-Based Machine Learning Classifier")

# Upload dataset files
train_file = st.file_uploader("Upload KDDTrain+ Data", type="txt")
test_file = st.file_uploader("Upload KDDTest+ Data", type="txt")

# Function to preprocess data and apply wavelet transform
def preprocess_data(filepaths, encoder=None, fit_encoder=False):

    dfs = [pd.read_csv(filepath, header=None) for filepath in filepaths]
    df = pd.concat(dfs, ignore_index=True)  # Combine both train and test files

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

# Preprocess and display results after clicking a button
if st.button("Process and Train Model") and train_file and test_file:
    # Preprocess both train and test files combined
    X, y, encoder = preprocess_data([train_file, test_file], fit_encoder=True)

    # Split combined data into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Classifier Selection Dropdown
    classifier_name = st.selectbox("Choose Classifier", ["RandomForest", "DecisionTree"])

    # Initialize classifiers with default parameters
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'DecisionTree': DecisionTreeClassifier(),
    }

    # Train and evaluate the selected classifier
    if classifier_name in classifiers:
        clf = classifiers[classifier_name]

        # Start CodeCarbon tracker (optional)
        # tracker = EmissionsTracker()
        # tracker.start()

        clf.fit(X_train_kbest, y_train)

        # Stop CodeCarbon tracker and display emissions (optional)
        # emissions = tracker.stop()
        # st.write(f"Energy consumed: {emissions:.6f} kWh")

        # Evaluate on test data
        y_test_pred = clf.predict(X_test_kbest)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)

        # Display results
        st.write(f"Test Accuracy: {accuracy_test:.7f}")
        st.write("Confusion Matrix:")
        st.write(cm_test)

        # Plot confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
                    yticklabels=np.unique(y_test), ax=ax)
        plt.title(f"Confusion Matrix - {classifier_name}")
        st.pyplot(fig)

        # Print classification report
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_test_pred, zero_division=0))
