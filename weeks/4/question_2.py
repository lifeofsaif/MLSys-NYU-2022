import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

def get_pipeline(df):
    numeric_cols = df.select_dtypes(include='number').columns.drop('id')
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.drop('satisfaction')

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler())
        ]), numeric_cols),
        
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('clf', LogisticRegression())
    ])

    return pipeline

def accuracy_optimized_grid_search(pipeline, X, y):
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100],  # regularization strength
        'clf__penalty': ['l2'],
        'clf__solver': ['lbfgs']  # make sure it supports l2
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid.fit(X, y)

    print("== Accuracy optimized grid search ==")
    print("Best parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

def roc_optimized_grid_search(pipeline, X, y):
    param_grid = {
        'clf__C': [0.1, 1, 10],
        'clf__solver': ['lbfgs', 'liblinear'],
        'clf__fit_intercept': [True, False]
    }

    # Split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # GridSearch with 5-fold CV on training set
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)

    print("== ROC optimized grid search ==")
    print("Best parameters:", grid.best_params_)
    print("Best cross-val AUC:", grid.best_score_)

    y_probs = grid.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)
    print("Test AUC:", auc)

def main():
    df = pd.read_csv("./data/airline_satisfaction/train.csv")
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    pipeline = get_pipeline(df)

    X = df.drop(columns='satisfaction')
    y = df['satisfaction']
    
    accuracy_optimized_grid_search(pipeline, X, y)
    # Output:
    # == Accuracy optimized grid search ==
    # Best parameters: {'clf__C': 0.01, 'clf__penalty': 'l2', 'clf__solver': 'lbfgs'}
    # Best score: 0.8750866539403382
    
    roc_optimized_grid_search(pipeline, X, y)
    # Output:
    # Best parameters: {'clf__C': 0.1, 'clf__fit_intercept': True, 'clf__solver': 'lbfgs'}
    # Best cross-val AUC: 0.9264035439540258
    # Test AUC: 0.9281039802012168

main()
    