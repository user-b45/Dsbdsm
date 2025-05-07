import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# 1. List Features and Their Types
print("🔹 Feature List and Types:\n")
for col in df.columns:
    dtype = df[col].dtype
    ftype = "Nominal" if col == 'species' else "Numeric"
    print(f"{col}: {dtype} → {ftype}")
print("\n")

# 2. Create Histograms for Each Feature
df.drop('species', axis=1).hist(figsize=(10, 6), bins=15, color='skyblue', edgecolor='black')
plt.suptitle("📊 Histograms of Numeric Features", fontsize=14)
plt.tight_layout()
plt.show()

# 3. Create Boxplots for Each Feature
plt.figure(figsize=(12, 8))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[column], color="lightgreen")
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()

# 4. Observations
print("📝 Observations and Inference:")
print("""
1. All features (sepal length, sepal width, petal length, petal width) are numeric.
2. 'Species' is nominal (categorical).
3. Histograms show:
   - Petal length and width are good separators between species.
   - Sepal width has a relatively uniform distribution.
4. Boxplots reveal:
   - Outliers may exist in 'sepal width' (a few low values).
   - Petal length and width show distinct clusters across species.
   - Variability in sepal length is moderate across all species.
""")

