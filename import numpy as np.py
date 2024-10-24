import numpy as np
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
import time
import scipy.stats as stats
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
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

# Load and preprocess training data from KDDTrain+
X_train, y_train, encoder = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+.txt", fit_encoder=True)

# Load and preprocess test data from KDDTest+
X_test, y_test, _ = preprocess_data("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTest+.txt", encoder=encoder)

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

# WGAN for Data Augmentation
def build_generator(noise_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=noise_dim),
        layers.Dense(256, activation='relu'),
        layers.Dense(output_dim, activation='tanh')
    ])
    return model

def build_critic(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_dim=input_dim),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    return model

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

# Training WGAN
def train_wgan(X_train, noise_dim=100, batch_size=64, epochs=10000):
    # Initialize generator and critic
    generator = build_generator(noise_dim, X_train.shape[1])
    critic = build_critic(X_train.shape[1])

    # Optimizers
    generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    critic_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

    # Compile critic
    critic.compile(optimizer=critic_optimizer, loss=wasserstein_loss)
    # Compile the generator before training
    generator.compile(optimizer=generator_optimizer, loss=wasserstein_loss)

    
    # Training loop
    for epoch in range(epochs):
        # Train the critic more than the generator
        for _ in range(5):
            # Random noise for the generator
            noise = np.random.normal(0, 1, (batch_size, noise_dim))

            # Real samples
            real_samples = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

            # Fake samples generated by the generator
            fake_samples = generator.predict(noise)

            # Label real as -1, fake as 1
            real_labels = -np.ones((batch_size, 1))
            fake_labels = np.ones((batch_size, 1))

            # Train critic on real and fake samples
            critic_loss_real = critic.train_on_batch(real_samples, real_labels)
            critic_loss_fake = critic.train_on_batch(fake_samples, fake_labels)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        misleading_labels = -np.ones((batch_size, 1))  # Train generator to make critic believe fake samples are real
        generator_loss = generator.train_on_batch(noise, misleading_labels)

        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Generator Loss: {generator_loss}, Critic Loss Real: {critic_loss_real}, Critic Loss Fake: {critic_loss_fake}")

    return generator

# Train WGAN and generate synthetic data
generator = train_wgan(X_train_kbest, noise_dim=100, epochs=5000)

# Generate synthetic data
noise = np.random.normal(0, 1, (1000, 100))  # Generate 1000 synthetic samples
synthetic_data = generator.predict(noise)

# Combine synthetic data with real training data
X_train_augmented = np.vstack([X_train_kbest, synthetic_data])
y_train_augmented = np.hstack([y_train, np.random.choice(y_train, 1000)])  # Assign random labels to synthetic data

# Initialize classifiers with default parameters
classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=100),
}

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train_augmented, y_train_augmented)

    # Evaluate on test data
    y_test_pred = clf.predict(X_test_kbest)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Plot confusion matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Test Confusion Matrix - {name}")
    plt.show()

    # Print accuracy and classification report
    print(f"Test Accuracy of {name}: {accuracy_test:.7f}")
    print(classification_report(y_test, y_test_pred, zero_division=0))

              
