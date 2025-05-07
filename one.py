import pandas as pd
import numpy as np

# Step 1: Load Dataset
try:
    df = pd.read_csv("autodata.csv")
    print("✅ Dataset successfully loaded.\n")
except FileNotFoundError:
    print("❌ Dataset file not found. Make sure 'StudentsPerformance.csv' is in the same directory.")
    exit()

# Step 2: Preview the data
print("📌 First 5 records:")
print(df.head(), "\n")

# Step 3: Check for missing values
print("📌 Missing Values:")
print(df.isnull().sum(), "\n")

# Step 4: Basic statistics
print("📌 Descriptive Statistics:")
print(df.describe(), "\n")

# Step 5: Dataset dimensions
print(f"📌 Dataset Shape: {df.shape}\n")

# Step 6: Data Types
print("📌 Data Types:")
print(df.dtypes, "\n")

# Step 7: Normalize numerical columns using Min-Max Scaling
df['math_score_norm'] = (df['math score'] - df['math score'].min()) / (df['math score'].max() - df['math score'].min())
df['reading_score_norm'] = (df['reading score'] - df['reading score'].min()) / (df['reading score'].max() - df['reading score'].min())
df['writing_score_norm'] = (df['writing score'] - df['writing score'].min()) / (df['writing score'].max() - df['writing score'].min())

print("✅ Normalization applied to math, reading, and writing scores.\n")

# Step 8: Encode categorical variables
df_encoded = pd.get_dummies(df, columns=[
    'gender', 
    'race/ethnicity', 
    'parental level of education', 
    'lunch', 
    'test preparation course'
], drop_first=True)

print("✅ Categorical variables converted to numerical (one-hot encoded).\n")

# Step 9: Show final DataFrame shape
print(f"📌 Final encoded DataFrame shape: {df_encoded.shape}")

# Optional: Save cleaned data
df_encoded.to_csv("cleaned_students_data.csv", index=False)
print("✅ Cleaned dataset saved as 'cleaned_students_data.csv'")
