import pandas as pd
from question_1 import get_95th_percentile_time_between_transaction_and_timestamp
from get_data import get_and_preprocess_data

# ===========
# Question 2
# ===========

# This question involves thinking through how to construct a training and test set 
# if you want to build an ML model to predict fraud. Remember, training and test sets 
# should approximate real world scenarios as much as possible.

# Imagine you train and deploy a model on the same day. You plan to collect a month's 
# worth of ground truth data in order to analyze the model's performance. 
# Construct a training and test dataset based on this scenario.

# Using the payments data, construct a test set containing a month's worth of 
# data and a training dataset containing as much data as possible. You should only use data in 
# the training dataset that you would know the ground truth for at the time of the start 
# of the test data. You can use your answer from part 1 help you determine which
# payments you would have solid ground truth for.

# In your script, divide the payments up into training and test data and print
# out the start and end timestamp of each dataset.

def get_train_test_split(df_payments, wait_days):
    test_start_date = pd.to_datetime('2022-08-01')  # example, adjust if needed
    test_end_date = test_start_date + pd.Timedelta(days=30)

    train_cutoff = test_start_date - pd.Timedelta(days=wait_days)

    # Train = payments that occurred before (test_start - wait buffer)
    train_df = df_payments[df_payments['transaction_timestamp'] <= train_cutoff]

    # Test = payments from test_start to test_end
    test_df = df_payments[
        (df_payments['transaction_timestamp'] >= test_start_date) &
        (df_payments['transaction_timestamp'] <= test_end_date)
    ].copy()
    return train_df, test_df

if __name__ == "__main__":
    df = get_and_preprocess_data()
    wait_days = get_95th_percentile_time_between_transaction_and_timestamp(df)
    train_df, test_df = get_train_test_split(df, wait_days)
    print(f"Train set start: {train_df['transaction_timestamp'].min()}")
    print(f"Train set end: {train_df['transaction_timestamp'].max()}")
    print(f"Test set start: {test_df['transaction_timestamp'].min()}")
    print(f"Test set end: {test_df['transaction_timestamp'].max()}")