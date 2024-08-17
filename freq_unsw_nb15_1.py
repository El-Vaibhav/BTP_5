import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.fft import fft

# Load the dataset
file_path = 'C:\\Users\\HP\\OneDrive\\Desktop\\ML\\UNSW-NB15_1.csv'
data = pd.read_csv(file_path, nrows=100000, header=None)

# Convert to frequency domain
def time_to_frequency(data):
    freq_data = np.abs(fft(data, axis=0))
    return pd.DataFrame(freq_data)

# Example columns to apply FFT
columns_to_transform = [6, 7, 8, 9, 10]  # Update with actual numeric columns if necessary

# Apply FFT to selected columns
for col in columns_to_transform:
    data[col] = time_to_frequency(data[[col]]).values

# Handle missing values and categorical data
data = data.fillna(0)
data = pd.get_dummies(data)

# Drop non-numeric columns if any
# Example column names to drop; update based on actual columns
drop_columns = [47]  # Example columns to drop; adjust as needed
data = data.drop(columns=drop_columns, errors='ignore')

# Define features and target
X = data.drop(columns=['label'], errors='ignore')  # Update with actual target column name
y = data['label']  # Update with actual target column name

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f'Accuracy: {accuracy:.4f}')
