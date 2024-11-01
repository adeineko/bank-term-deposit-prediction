import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Load your dataset
df = pd.read_csv("../dataset/updated_bank.csv")

# Prepare the features and target variable
X = df.drop("y", axis=1)
y = df["y"]

# Encode categorical variables in features
label_enc = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_enc.fit_transform(X[column])

# Encode the target variable
y = LabelEncoder().fit_transform(y)  # Convert 'yes'/'no' to 1/0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base models
base_models = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('svc', SVC(probability=True)),  # SVM
    ('knn', KNeighborsClassifier()),  # KNN
    ('xgb', XGBClassifier(eval_metric='logloss'))  # XGBoost
]

# Define the meta model
meta_model = LogisticRegression()

# Create the stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Hyperparameter tuning
param_grid = {
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [None, 10, 15],
    'gb__n_estimators': [50, 100, 150],
    'gb__learning_rate': [0.01, 0.1, 0.2],
    'svc__C': [0.1, 1, 2],
    'knn__n_neighbors': [3, 5, 7],
    'xgb__max_depth': [3, 5, 7],
    'xgb__n_estimators': [50, 100],
}

# Use f1 score for scoring (ensure y is now binary)
grid_search = GridSearchCV(estimator=stacking_clf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)

# Fit the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)
