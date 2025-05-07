# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
# Make sure to download the 'boston.csv' file from Kaggle and place it in the correct directory
df = pd.read_csv('HousingData.csv')  # adjust the path if needed

# Step 2: Understanding the dataset
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize the correlation matrix of features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Step 3: Define the feature variables (X) and target variable (y)
X = df.drop('MEDV', axis=1)  # MEDV is the target (price)
y = df['MEDV']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = lr.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Step 8: Visualize the prediction vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='crimson')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'b--')  # line for perfect predictions
plt.grid()
plt.show()
