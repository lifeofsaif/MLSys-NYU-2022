import pandas as pd

def get_and_preprocess_data():
    df_payments = pd.read_csv('data/payments.csv')
    df_payments['transaction_timestamp'] = pd.to_datetime(df_payments['transaction_timestamp'], errors='coerce')
    df_payments['chargeback_timestamp'] = pd.to_datetime(df_payments['chargeback_timestamp'], errors='coerce')

    df_payments['time_between_transaction_and_chargeback'] = (df_payments['chargeback_timestamp'] - df_payments['transaction_timestamp']).dt.days

    df_payments = df_payments.sort_values('transaction_timestamp')
    
    df_payments['is_fraud'] = df_payments['chargeback_timestamp'].notna().astype(int)

    df_buyers = pd.read_csv('./data/buyers.csv')
    df_merchants = pd.read_csv('./data/merchants.csv')

    # merge data from merchants to payments
    df_merchants = df_merchants.rename(columns={'id': 'merchant_id', 'country': 'merchant_country', 'category': 'merchant_category'})
    df_payments = df_payments.merge(df_merchants[['merchant_country', 'merchant_category', 'merchant_id']], on='merchant_id', how='left')

    # merge data from buyers to payments
    df_buyers = df_buyers.rename(columns={'id': 'buyer_id', 'country': 'buyer_country'})
    df_payments = df_payments.merge(df_buyers[['buyer_country', 'buyer_id']], on='buyer_id', how='left')


    return df_payments