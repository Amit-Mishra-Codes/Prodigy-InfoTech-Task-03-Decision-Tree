import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_PATH = "data/bank-full.csv"
OUTPUT_DIR = "output"

df = pd.read_csv(DATA_PATH, sep=";")
X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols),
])

model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

with open(f"{OUTPUT_DIR}/model_metrics.txt", "w", encoding="utf-8") as f:
    f.write("Prodigy InfoTech Task 3 - Decision Tree Classifier\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples: {len(X_test)}\n\n")
    f.write(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=["No", "Yes"], zero_division=0))

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["No", "Yes"])
plt.yticks([0, 1], ["No", "Yes"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=200)
plt.close()

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["classifier"].feature_importances_
top = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
plt.figure(figsize=(9, 6))
top.sort_values().plot(kind="barh")
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=200)
plt.close()

plt.figure(figsize=(22, 12))
plot_tree(pipeline.named_steps["classifier"], feature_names=feature_names,
          class_names=["No", "Yes"], max_depth=3, filled=True, fontsize=7)
plt.title("Decision Tree Classifier (First 3 Levels)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/decision_tree.png", dpi=200)
plt.close()

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
