import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate Synthetic Student Data (No Internet / Download Needed)
np.random.seed(42)
n_samples = 300

data = {
    'studytime': np.random.randint(1, 5, n_samples),
    'failures': np.random.randint(0, 4, n_samples),
    'famrel': np.random.randint(1, 6, n_samples),
    'freetime': np.random.randint(1, 6, n_samples),
    'goout': np.random.randint(1, 6, n_samples),
    'absences': np.random.randint(0, 30, n_samples),
    'G1': np.random.randint(4, 20, n_samples),
    'G2': np.random.randint(4, 20, n_samples),
}

df = pd.DataFrame(data)

# Calculate G3 grade logically based on performance factors
df['G3'] = (df['G1'] * 0.4 + df['G2'] * 0.5 - df['failures'] * 1.5 + df['studytime'] * 0.8).round().astype(int)
df['G3'] = df['G3'].clip(lower=0, upper=20)

# Target: 1 = Safe (Pass), 0 = At-Risk (Fail)
df['target'] = (df['G3'] >= 10).astype(int)

# 2. Features and Target Selection
features = ['studytime', 'failures', 'famrel', 'freetime', 'goout', 'absences', 'G1', 'G2']
X = df[features]
y = df['target']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Models & Compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

print("--- STUDENT PERFORMANCE ML RESULTS ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_test, preds) * 100, 2), "%")
    print(classification_report(y_test, preds))