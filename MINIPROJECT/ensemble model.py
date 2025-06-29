import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
import joblib

# 1️⃣ **Load Dataset**
df = pd.read_csv('smote.csv')
X = df.drop(columns=['Heart_Disease_Status'])  
y = df['Heart_Disease_Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 2️⃣ **Define Hyperparameter Grids**
param_grids = {
    'log_reg': {
        'C': np.logspace(-3, 3, 3),
        'penalty': ['l1', 'l2']
    },
    'random_forest': {
        'n_estimators': [100, 300],
        'max_depth': [10, 30, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
        'max_features': ['sqrt', 'log2']
    },
    'naive_bayes': {},  # No hyperparameters to tune for GaussianNB
    'xgboost': {
        'n_estimators': [100, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'lightgbm': {
        'n_estimators': [100, 300],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50],
        'max_depth': [-1, 10],
        'subsample': [0.8, 1.0]
    }
}

# 3️⃣ **Initialize Base Models**
models = {
    'log_reg': LogisticRegression(solver='saga', max_iter=500, random_state=42),
    'random_forest': RandomForestClassifier(random_state=42),
    'naive_bayes': GaussianNB(),
    'xgboost': XGBClassifier(eval_metric='logloss', random_state=42),
    'lightgbm': LGBMClassifier(random_state=42)
}

# 4️⃣ **Hyperparameter Tuning for Each Model**
best_models = {}

for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    param_grid = param_grids[model_name]
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=5,
        scoring='accuracy',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_name}: {random_search.best_params_}")
    print(f"Best score for {model_name}: {random_search.best_score_}")
    
    best_models[model_name] = random_search.best_estimator_

# 5️⃣ **Voting Classifier**
voting_clf = VotingClassifier(
    estimators=[
        ('log_reg', best_models['log_reg']),
        ('random_forest', best_models['random_forest']),
        ('naive_bayes', best_models['naive_bayes']),
        ('xgboost', best_models['xgboost']),
        ('lightgbm', best_models['lightgbm'])
    ],
    voting='soft'
)

# 6️⃣ **Train the Voting Classifier**
voting_clf.fit(X_train, y_train)

# 7️⃣ **Evaluate on the Test Set**
y_pred = voting_clf.predict(X_test)
print("\nClassification Report for Voting Classifier:\n")
print(classification_report(y_test, y_pred))

# 8️⃣ **Save the Final Ensemble Model**
joblib.dump(voting_clf, 'ensemble_model_with_tuning.joblib')
print("\nEnsemble model saved as 'ensemble_model_with_tuning.joblib'\n")