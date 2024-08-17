import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical

# Read the dataset

df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\data.txt", header=None)

# Drop non-numeric columns, keep the last column for labels
df_numeric = df.drop(columns=[1, 2, 3, df.columns[-1]])

# Convert remaining columns to float
df_numeric = df_numeric.astype(float)

# Apply windowing to reduce spectral leakage
window = windows.hamming(df_numeric.shape[1])
data_numeric = df_numeric.values * window

# Extract labels
labels = df[df.columns[-1]]

# Convert numeric data to numpy array
data_numeric = np.array(data_numeric)

# Apply FFT along each row (axis=1)
fft_data = np.fft.fft(data_numeric, axis=1)

# Take the magnitude of the FFT results
fft_magnitude = np.abs(fft_data)

# Reshape the data for LSTM
X = fft_magnitude[:, :, np.newaxis]  # Add a new axis for features ( time_steps = number of columns )

# Convert labels to categorical
label_mapping = {label: idx for idx, label in enumerate(labels.unique())}
y = np.array([label_mapping[label] for label in labels])
y = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_classes, y_pred_classes)
cm = confusion_matrix(y_test_classes, y_pred_classes)

print(f"Confusion Matrix for LSTM:")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.title(f"Confusion Matrix - LSTM")
plt.show()

# Print accuracy and classification report
print(f"Accuracy of LSTM: {accuracy:.7f}")
print(classification_report(y_test_classes, y_pred_classes, zero_division=0))
