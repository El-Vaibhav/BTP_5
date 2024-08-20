import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, windows
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical

# Function to preprocess data with Hilbert Transform
def preprocess_data_hilbert(filepath):
    df = pd.read_csv(filepath, header=None)
    df_numeric = df.drop(columns=[1, 2, 3, df.columns[-2]])
    df_numeric = df_numeric.astype(float)
    window = windows.hamming(df_numeric.shape[1])
    data_numeric = df_numeric.values * window
    labels = df[df.columns[-2]]
    data_numeric = np.array(data_numeric)
    hilbert_transformed = np.abs(hilbert(data_numeric, axis=1))
    X = hilbert_transformed[:, :, np.newaxis]
    label_mapping = {label: idx for idx, label in enumerate(labels.unique())}
    y = np.array([label_mapping[label] for label in labels])
    y = to_categorical(y, num_classes=len(label_mapping))
    return X, y, len(label_mapping)

# Load and preprocess training data
X_train, y_train, num_classes = preprocess_data_hilbert("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTrain+.txt")

# Load and preprocess test data
X_test, y_test, _ = preprocess_data_hilbert("C:\\Users\\HP\\OneDrive\\Desktop\\BTP_5thsem\\BTP\\KDDTest+.txt")

# Verify number of classes
print(f"Number of classes: {num_classes}")

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(23, activation='softmax'))  # Update the number of units
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate on training data
y_train_pred = model.predict(X_train)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)
y_train_classes = np.argmax(y_train, axis=1)
accuracy_train = accuracy_score(y_train_classes, y_train_pred_classes)
cm_train = confusion_matrix(y_train_classes, y_train_pred_classes)

# Evaluate on test data
y_test_pred = model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
accuracy_test = accuracy_score(y_test_classes, y_test_pred_classes)
cm_test = confusion_matrix(y_test_classes, y_test_pred_classes)

# Print accuracy and classification report for test data
print(f"Confusion Matrix for LSTM on Test Data:")
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - LSTM (Test Data)")
plt.show()
print(f"Accuracy of LSTM on Test Data: {accuracy_test:.7f}")
print(classification_report(y_test_classes, y_test_pred_classes, zero_division=0))


