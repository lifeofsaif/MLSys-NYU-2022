'''
Stream lit app - loads latest flow and allows user to select some performance results from a dropdown
'''

from sklearn.metrics import accuracy_score
import streamlit as st
from metaflow import Flow, metadata, get_metadata

FLOW_NAME = 'LogisticRegressionFlow'
metadata('./')
print(get_metadata())

# build up the dashboard
st.markdown("# Regression playground")
st.write("This application shows the dataset and predictions made by our model!")

def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r

def get_slice_accuracy(column_name, y_pred, y_test, X_test):
    mask = X_test[column_name] > 0
    return accuracy_score(y_pred[mask], y_test[mask])
        
# get artifacts from latest run, using Metaflow Client API
latest_run = get_latest_successful_run(FLOW_NAME)
latest_model = latest_run.data.model
latest_X_train = latest_run.data.X_train
latest_X_test = latest_run.data.X_test
y_pred = latest_run.data.y_pred
y_test = latest_run.data.y_test

# show dataset
st.markdown("## Dataset")
st.write("First 10 Xs in the training set:")
st.write(latest_X_train[:10])

# play with model
st.markdown("### See the accuracy for a given slice")

column_name_1 = st.selectbox("slice1", latest_X_test.columns)
st.markdown('Accuracy for slice1 {}: **{}**'.format(column_name_1, get_slice_accuracy(column_name_1, y_pred, y_test, latest_X_test)))
        
column_name_2 = st.selectbox("slice2", latest_X_test.columns)
st.markdown('Accuracy for slice2 {}: **{}**'.format(column_name_2, get_slice_accuracy(column_name_2, y_pred, y_test, latest_X_test)))
