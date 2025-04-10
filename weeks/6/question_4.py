from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from get_data import get_and_preprocess_data
from question_1 import get_95th_percentile_time_between_transaction_and_timestamp
from question_2 import get_train_test_split
from question_3 import get_averages_rolling_groupby

# ===============
# Question 4
# ===============

# For the final question, you should build an ML model to predict fraud. 
# Your model should join together the payments, merchant, and buyer data and use the following features:

# payment_amount
# merchant category
# merchant country
# buyer country
# merchant fraud rate (from question 3)
# buyer fraud rate (from question 3)

# Split your data up into training and test data using your answers from question 2, 
# train a model on the training data, and evaluate the model's area under the ROC curve (AUC) on the test data. 
# Print out the AUC at the end. You model will likely perform poorly, and that's ok! Fraud is hard.

def get_pipeline(df):
    numeric_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns

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
        ('clf', LogisticRegression(max_iter=1000))
    ])

    return pipeline

def roc_optimized_grid_search(pipeline, X_train, y_train, X_test, y_test):
    param_grid = {
        'clf__C': [0.1, 1, 10],
        'clf__solver': ['lbfgs', 'liblinear'],
        'clf__fit_intercept': [True, False]
    }

    # GridSearch with 5-fold CV on training set
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)

    print("== ROC optimized grid search ==")
    print("Best parameters:", grid.best_params_)
    print("Best cross-val AUC:", grid.best_score_)

    y_probs = grid.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)
    print("Test AUC:", auc)



# 1. Load the data and merge the data sets
df_payments = get_and_preprocess_data()

# 2. Split the data into training and test sets

wait_days = get_95th_percentile_time_between_transaction_and_timestamp(df_payments)
train_df, test_df = get_train_test_split(df_payments, wait_days)

train_df = get_averages_rolling_groupby(train_df)

merchant_rates = train_df.groupby('merchant_id')['average_merchant_fraud_rate'].last()
buyer_rates = train_df.groupby('buyer_id')['average_buyer_fraud_rate'].last()

test_df['average_merchant_fraud_rate'] = test_df['merchant_id'].map(merchant_rates).fillna(0)
test_df['average_buyer_fraud_rate'] = test_df['buyer_id'].map(buyer_rates).fillna(0)

train_X = train_df.drop(columns=['is_fraud', 'transaction_timestamp', 'chargeback_timestamp'])
train_y = train_df['is_fraud']

test_X = test_df.drop(columns=['is_fraud', 'transaction_timestamp', 'chargeback_timestamp'])
test_y = test_df['is_fraud']

# 3. Train the model on the training data

pipeline = get_pipeline(train_X)
roc_optimized_grid_search(pipeline, train_X, train_y, test_X, test_y)
