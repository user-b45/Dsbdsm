# 1. Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 2. Create a sample "Academic Performance" dataset
data = {
    'Student_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'Math_Score': [88, 92, 79, np.nan, 85, 95, 100, 20, 88, 89],
    'Science_Score': [91, 85, 76, 82, 45, 78, np.nan, 35, 88, 87],
    'Attendance(%)': [90, 80, 85, 75, 95, 100, 30, 25, 85, 90]
}

df = pd.DataFrame(data)

# Display original dataset
print("Original Dataset:")
print(df)

# 3. Handle Missing Values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing numeric values with mean
df['Math_Score'].fillna(df['Math_Score'].mean(), inplace=True)
df['Science_Score'].fillna(df['Science_Score'].mean(), inplace=True)

# 4. Detect Outliers using Z-score method
numeric_cols = ['Math_Score', 'Science_Score', 'Attendance(%)']

z_scores = np.abs(stats.zscore(df[numeric_cols]))
outliers = (z_scores > 3)

print("\nOutliers detected:")
print(outliers)

# Optionally, cap/floor outliers using IQR (Interquartile Range)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] > upper_bound, upper_bound,
                np.where(df[col] < lower_bound, lower_bound, df[col]))

# 5. Apply Data Transformation (Log transformation on Attendance)
# Reason: Attendance has a right-skewed distribution due to low attendance values (e.g., 25, 30%)
# Goal: Reduce skewness and make it more normal

# Add 1 to avoid log(0)
df['Log_Attendance'] = np.log(df['Attendance(%)'] + 1)

# Display transformed data
print("\nTransformed Dataset:")
print(df[['Attendance(%)', 'Log_Attendance']])

# Optional: Visual comparison of original vs. transformed
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
sns.histplot(df['Attendance(%)'], kde=True)
plt.title('Original Attendance Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['Log_Attendance'], kde=True)
plt.title('Log-Transformed Attendance Distribution')

plt.tight_layout()
plt.show()
