import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Display basic structure
print("📊 Titanic Dataset Sample:")
print(titanic[['sex', 'age', 'survived']].dropna().head(), "\n")

# Box plot of age vs sex and survival
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=titanic, palette='Set2')

plt.title('📦 Age Distribution by Gender and Survival')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

# Observations
print("\n📝 Observations:")
print("""
1. Female passengers had a broader age range among survivors compared to males.
2. Among males, survivors were generally younger; older males had lower survival.
3. Very young passengers (infants/children) had higher survival probability.
4. Females, in general, had higher survival rates, showing the 'women and children first' policy.
""")

