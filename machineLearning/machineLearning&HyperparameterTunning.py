import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.svm import SVC

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("../dataset/updated_bank.csv")

X = df.drop("y", axis=1)
y = df["y"]

label_enc = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_enc.fit_transform(X[column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

gb_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=5, scoring='accuracy')
gb_grid_search.fit(X_train, y_train)

# Best Gradient Boosting model
best_gb_model = gb_grid_search.best_estimator_
print("Best parameters for Gradient Boosting: ", gb_grid_search.best_params_)

# Model evaluation for the best Gradient Boosting model
y_pred_gb = best_gb_model.predict(X_test)
conf_matrix_gradient_boosting = confusion_matrix(y_test, y_pred_gb)
report_gradient_boosting = classification_report(y_test, y_pred_gb)

print("Confusion matrix for best Gradient Boosting: \n", conf_matrix_gradient_boosting)
print("Report for best Gradient Boosting: \n", report_gradient_boosting)

# Define base models for stacking with additional models
base_models = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', best_gb_model),  # Using the best Gradient Boosting model found
    ('svc', SVC(probability=True, random_state=42)),  # Support Vector Classifier
    ('knn', KNeighborsClassifier()),  # K-Nearest Neighbors
    ('xgb', XGBClassifier(random_state=42)),  # XGBoost Classifier
    ('et', ExtraTreesClassifier(random_state=42))  # Extra Trees Classifier
]

# Define the meta-model (final estimator)
meta_model = LogisticRegression()

# Create the StackingClassifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Hyperparameter tuning for Stacking model
stacking_param_grid = {
    'final_estimator__C': [0.01, 0.1, 1.0, 10.0],  # Regularization strength for Logistic Regression
    'final_estimator__max_iter': [100, 200, 300]
}

stacking_grid_search = GridSearchCV(stacking_model, stacking_param_grid, cv=5, scoring='accuracy')
stacking_grid_search.fit(X_train, y_train)

# Best Stacking model
best_stacking_model = stacking_grid_search.best_estimator_
print("Best parameters for Stacking: ", stacking_grid_search.best_params_)

# Make predictions and evaluate the best Stacking model
y_pred_stack = best_stacking_model.predict(X_test)
conf_matrix_stacking = confusion_matrix(y_test, y_pred_stack)
report_stacking = classification_report(y_test, y_pred_stack)

print("Confusion matrix for best Stacking: \n", conf_matrix_stacking)
print("Report for best Stacking: \n", report_stacking)