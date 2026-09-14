# Prodigy InfoTech Task 3 - Decision Tree Classifier

## Objective
Build a decision tree classifier to predict whether a customer will purchase/subscribe to a product or service based on demographic and behavioral data.

## Dataset
Bank Marketing dataset from the UCI Machine Learning Repository, provided through the official Prodigy InfoTech Task 3 dataset repository.

The full dataset contains 45,211 records and 17 columns. The target variable is `y`, where `yes` means the client subscribed to a term deposit and `no` means the client did not.

## Technologies
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Methodology
1. Load the semicolon-separated Bank Marketing dataset.
2. Separate the target variable `y` from the input features.
3. Convert the target to binary values (`yes = 1`, `no = 0`).
4. One-hot encode categorical variables.
5. Split the data into 80% training and 20% testing sets using stratification.
6. Train a Decision Tree Classifier with `max_depth=5`, `class_weight="balanced"`, and `random_state=42`.
7. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
8. Visualize feature importance and the first three levels of the decision tree.

## Project Structure
```text
Prodigy-InfoTech-Task-03-Decision-Tree/
├── data/
│   └── bank-full.csv
├── output/
│   ├── confusion_matrix.png
│   ├── decision_tree.png
│   ├── feature_importance.png
│   └── model_metrics.txt
├── decision_tree_classifier.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python decision_tree_classifier.py
```

## Note
The dataset includes `duration`, the length of the last contact. It is useful for demonstrating the internship classification task, but in a real pre-call prediction system it would not be available before the call and could cause target leakage.

## Internship
Data Science Internship - Prodigy InfoTech

## Author
Amit Mishra  
B.Sc. Data Science and Data Analytics
