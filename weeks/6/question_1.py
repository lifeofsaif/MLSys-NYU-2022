import pandas as pd
from get_data import get_and_preprocess_data

# ===========
# Question 1
# ===========

# In order to properly train and test a model, you must ensure that you're only using information to 
# train the model that you would have at the time that you are predicting with the model.

# # For the Fraud dataset, we assume that if there is no chargeback then the payment was 
# not fraudulent. However, a payment for yesterday may have a chargeback tomorrow. 
# This means that today I will think it's good, whereas tomorrow I'll know it's fraudulent. 
# Using the dataset, determine how many days you must wait after a payment is processed in 
# order to be sure that at least 95% of payments which would have received a chargeback 
# will have received a chargeback.

# # Your script should read in the appropriate dataset(s) and print out the number of days.

def get_95th_percentile_time_between_transaction_and_timestamp(df_payments):
    # Calculate the 95th percentile of the time between transaction and chargeback
    # 31
    wait_days = df_payments['time_between_transaction_and_chargeback'].quantile(0.95)

    return wait_days

if __name__ == "__main__":
    df = get_and_preprocess_data()
    wait_days = get_95th_percentile_time_between_transaction_and_timestamp(df)
    print(f"Wait days: {wait_days}")
