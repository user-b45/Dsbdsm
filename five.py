import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Step 1: Load the dataset
try:
    df = pd.read_csv("Social_Network_Ads.csv")
    print("✅ Dataset loaded successfully.\n")
except FileNotFoundError:
    print("❌ File not found. Please ensure 'Social_Network_Ads.csv' is in the same folder.")
    exit()

# Step 2: Select features and target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']  # 0 or 1

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

# Step 4: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
print("✅ Logistic Regression model trained.\n")

# Step 5: Predict on test set
y_pred = model.predict(X_test)

# Step 6: Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("📊 Confusion Matrix:")
print(cm)
print(f"\nTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

# Step 7: Compute Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy:.2f}")
print(f"❌ Error Rate: {error_rate:.2f}")
print(f"🎯 Precision: {precision:.2f}")
print(f"🔁 Recall: {recall:.2f}")

