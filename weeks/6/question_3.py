# ===========
# Quesrtion 3
# ===========

# For each payment, calculate the average fraud rate for the merchant and the buyer. 
# If there are no prior transactions, then assume that the rate is zero. 
# Note: your features should be calculated using only information that is known at the time of the transaction.

# In order to verify your answer, your script should print out the sum of 
# average merchant fraud rates for all payments plus the sum of average 
# buyer fraud rates for all payments.


# Do it iteratively ================
# This is a naive implementation and not efficient for large datasets.
import time

from get_data import get_and_preprocess_data


def get_averages_iterative(df_payments):
    df_payments['average_merchant_fraud_rate'] = 0
    df_payments['average_buyer_fraud_rate'] = 0

    for index, row in df_payments.iterrows():
        transaction_timestamp = row['transaction_timestamp']
        merchant_id = row['merchant_id']
        buyer_id = row['buyer_id']

        # Merchant
        merchant_txns = df_payments[
            (df_payments['merchant_id'] == merchant_id) &
            (df_payments['transaction_timestamp'] < transaction_timestamp)
        ]
        if len(merchant_txns) > 0:
            fraud_count = merchant_txns['time_between_transaction_and_chargeback'].notna().sum()
            df_payments.at[index, 'average_merchant_fraud_rate'] = fraud_count / len(merchant_txns)

        # Buyer
        buyer_txns = df_payments[
            (df_payments['buyer_id'] == buyer_id) &
            (df_payments['transaction_timestamp'] < transaction_timestamp)
        ]
        if len(buyer_txns) > 0:
            fraud_count = buyer_txns['time_between_transaction_and_chargeback'].notna().sum()
            df_payments.at[index, 'average_buyer_fraud_rate'] = fraud_count / len(buyer_txns)

    return df_payments


# Do it using rolling groupby ================
# This is a more efficient implementation using groupby and rolling operations.
def get_averages_rolling_groupby(df):
    df = df.sort_values('transaction_timestamp').reset_index(drop=True)

    df['merchant_transaction_count'] = (
        df.groupby('merchant_id', group_keys=False)
        .cumcount()
    )

    df['merchant_cum_fraud'] = (
        df.groupby('merchant_id', group_keys=False)['is_fraud']
        .apply(lambda x: x.cumsum().shift(1))
    )

    df['average_merchant_fraud_rate'] = (
        df['merchant_cum_fraud'] / df['merchant_transaction_count']
    ).fillna(0)

    df['buyer_transaction_count'] = (
        df.groupby('buyer_id', group_keys=False)
        .cumcount()
    )    

    df['buyer_cum_fraud'] = (
        df.groupby('buyer_id', group_keys=False)['is_fraud']
        .apply(lambda x: x.cumsum().shift(1))
    )

    df['average_buyer_fraud_rate'] = (
        df['buyer_cum_fraud'] / df['buyer_transaction_count']
    ).fillna(0)

    df = df.drop(['buyer_cum_fraud', 'merchant_cum_fraud'], axis=1)
    df = df.drop(['merchant_transaction_count', 'buyer_transaction_count'], axis=1)

    return df

if __name__ == "__main__":
    df_payments = get_and_preprocess_data()

    start_iter = time.time()
    df_payments_iterative = get_averages_iterative(df_payments)
    end_iter = time.time()
    print(f"Iterative version took: {end_iter - start_iter:.2f} seconds")
    print("Sum of avg merchant fraud rates:", df_payments_iterative['average_merchant_fraud_rate'].sum())
    print("Sum of avg buyer fraud rates:", df_payments_iterative['average_buyer_fraud_rate'].sum())

    start_roll = time.time()
    df_payments_rolling = get_averages_rolling_groupby(df_payments)
    end_roll = time.time()
    print(f"Rolling version took: {end_roll - start_roll:.2f} seconds")
    print("Sum of avg merchant fraud rates:", df_payments_rolling['average_merchant_fraud_rate'].sum())
    print("Sum of avg buyer fraud rates:", df_payments_rolling['average_buyer_fraud_rate'].sum())