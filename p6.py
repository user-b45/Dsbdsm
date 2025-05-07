# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the Iris dataset
df = pd.read_csv('Iris.csv')  # adjust path if needed
print("First few rows of the dataset:\n", df.head())

# Step 2: Preprocess data (drop 'Id' column and define features and target variable)
df = df.drop(columns=['Id'])
X = df.drop(columns=['Species'])  # Features
y = df['Species']  # Target

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train a Naive Bayes model (GaussianNB is used for continuous features)
nb = GaussianNB()
nb.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = nb.predict(X_test)

# Step 6: Compute Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display confusion matrix
print("\nConfusion Matrix:\n", conf_matrix)

# Step 7: Compute Accuracy, Precision, Recall, F1 Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Step 8: Compute Error Rate
error_rate = 1 - accuracy

# Display the computed metrics
print("\nModel Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Step 9: Visualize Confusion Matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=df['Species'].unique(), yticklabels=df['Species'].unique())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
