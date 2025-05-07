import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# 1. Check basic structure of Titanic dataset
print("📊 Titanic Dataset Info:")
print(titanic.info(), "\n")

# 2. Plot Histogram for 'fare' column (Ticket Price distribution)
plt.figure(figsize=(10, 6))
sns.histplot(titanic['fare'], kde=True, bins=30, color='skyblue', edgecolor='black')

plt.title('🎟️ Distribution of Ticket Fare on the Titanic')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

