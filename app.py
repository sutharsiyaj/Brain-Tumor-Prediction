!pip install pandas matplotlib scikit-learn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv("brain_cancer_data.csv")
print("Dataset Loaded Successfully")
print(df)
# Input features

X = df[["Age", "Headache", "MRI_Score", "Tumor_Size"]]

# Output label

y = df["Cancer"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()

model.fit(X_train, y_train)

print("Model Trained Successfully")
predictions = model.predict(X_test)

print("Predictions:")
print(predictions)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
df["Cancer"].value_counts().plot(kind="bar")

plt.title("Cancer vs Normal Patients")
plt.xlabel("Condition")
plt.ylabel("Count")

plt.show()