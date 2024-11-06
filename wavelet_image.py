import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Function to preprocess data and apply wavelet transform
def preprocess_data(filepath, encoder=None, fit_encoder=False, plot_wavelet=False):
    df = pd.read_csv(filepath, header=None)

  
    num_rows = int(len(df) * 0.02)

    df = df.iloc[:num_rows]

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
    additional_features = []

    for row in data_numeric:
        # Apply wavelet transform
        coeffs = pywt.wavedec(row, 'db4', level=4)  # Use 'db4' wavelet
        coeffs_flat = np.hstack(coeffs)

        # Append wavelet coefficients
        wavelet_coeffs.append(coeffs_flat)

        # Additional features
        mean_coeffs = np.mean(coeffs_flat)
        std_dev_coeffs = np.std(coeffs_flat)
        var_coeffs = np.var(coeffs_flat)
        max_coeffs = np.max(coeffs_flat)
        min_coeffs = np.min(coeffs_flat)
        range_coeffs = max_coeffs - min_coeffs
        mad_coeffs = np.mean(np.abs(coeffs_flat - mean_coeffs))

        additional_features.append([
            mean_coeffs, std_dev_coeffs, var_coeffs, max_coeffs, min_coeffs, range_coeffs, mad_coeffs
        ])

    # Convert wavelet coefficients and additional features to DataFrames
    wavelet_coeffs_df = pd.DataFrame(wavelet_coeffs)
    additional_features_df = pd.DataFrame(additional_features, columns=[
        "Mean", "StdDev", "Variance", "Max", "Min", "Range", "MAD"
    ])

    if plot_wavelet:
        plot_wavelet_coefficients(wavelet_coeffs_df.values, labels)

    return wavelet_coeffs_df, additional_features_df, labels, encoder

# Function to plot wavelet coefficients with labels
def plot_wavelet_coefficients(wavelet_data, labels):
    unique_labels = labels.unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # Generate distinct colors
    label_color_map = dict(zip(unique_labels, colors))
    
    plt.figure(figsize=(12, 6))

    for i, (row, label) in enumerate(zip(wavelet_data, labels)):
        # Plot the coefficient magnitudes with the color corresponding to the label
        plt.plot(np.arange(len(row)), np.abs(row),
                 color=label_color_map[label], label=label if i == 0 else "")
    
    # Add legend (one entry per label)
    handles = [plt.Line2D([0], [0], color=color, label=label) for label, color in label_color_map.items()]
    plt.legend(handles=handles, loc="upper right")
    
    plt.yscale("log")  # Optional: Log scale for better visibility
    plt.xlabel("Coefficient Index")
    plt.ylabel("Magnitude")
    plt.title("Wavelet Coefficients by Label")
    plt.show()

# Usage
filepath = "C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+_20Percent.txt"
wavelet_data, additional_features, labels, encoder = preprocess_data(
    filepath, fit_encoder=True, plot_wavelet=True
)
