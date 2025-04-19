'''
# Homework #4 (due 11/04/2022)

## Task 1: Experiment tracking (with Comet) and hyperparameter tuning (75%)
Thanks to our partnership with Comet ML, we can use their experiment tracking tool for free! We discussed the importance of experiment tracking in the class, and in this assignment you will be asked to start doing it! Consider again [small_flow.py](https://github.com/jacopotagliabue/MLSys-NYU-2022/blob/main/weeks/8/src/small_flow.py)  (the simple DAG-based regression training).

 Duplicate the script, change the class name, and change the `train_model` step: instead of a linear regression, use any other scikit-available algorithm from the class, as long as the algorithm has one hyperparameter (make sure to use a dataset that makes sense with the algorithm you choose - if you choose a classification algorithm, you need to generate an appropriate synthetic dataset). 
 
 Instead of training one model, you will:

* Make sure to have a validation split, not just train and test split.
* Use metaflow parallelization capabilities (foreach or branch!) to train multiple models: pick 4 values for the hyperparameter and train the model 4 times, each time with a different value.
* Evaluate the models on the validation set and pick the best one.
* Make sure to log the experiments in Comet: does the dashboard clearly show what model is the best?

When you submit the code, submit both the new script and screenshot / prints from your Comet dashboard, clearly illustrating the tracking (you can also share the dashboard with your TA)

Useful links: [Comet scikit docs](https://www.comet.com/docs/v2/integrations/ml-frameworks/scikit-learn/)
, [Metaflow foreach docs](https://docs.metaflow.org/metaflow/basics).

'''

from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os
import numpy as np
import pandas as pd
from comet_ml import ExistingExperiment, Experiment
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'

assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']
assert 'MY_PROJECT_NAME' in os.environ and os.environ['MY_PROJECT_NAME']

class LogisticRegressionFlow(FlowSpec):
    """
    LogisticRegressionFlow is a minimal DAG showcasing reading data from a file,
    cleaning and prepping data, feature engineering, and then 
    """

    DATA_FILE = IncludeFile(
        'dataset',
        help='Census data from https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        is_text=True,
        default='data/data.csv'
    )

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20
    )
    
    @step
    def start(self):
      """
      Start up and print out some info to make sure everything is ok metaflow-side
      """
      print("Starting up at {}".format(datetime.utcnow()))
      # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
      # to show how information about the current run can be accessed programmatically
      print("flow name: %s" % current.flow_name)
      print("run id: %s" % current.run_id)
      print("username: %s" % current.username)
      self.next(self.load_data)
      
      
    @step
    def load_data(self): 
      """
      Read the data in from the static file
      """
      from io import StringIO
      
      df = pd.read_csv(StringIO(self.DATA_FILE), header=None)
      self.df = df
      self.next(self.prepare_data)
      
    @step 
    def prepare_data(self):
      """
      Prepare data. May be best to split this up later
      - adds column names
      - replaces ? with nan
      - one hot encodes 
      - drops any unused columns
      - 
      """
      print("Preparing data")
      df = self.df
      
      # add columns since columns are missing
      column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
      ]
      df.columns = column_names
      
      # this dataset has a ? for missing values, so replace it with nan
      df = df.replace(' ?', np.nan)
      
      # one hot encode categorical columns
      cat_cols = df.select_dtypes(include='object').columns.drop('income')
      df = pd.get_dummies(df, columns=cat_cols)
      
      # drop fnlwt for now. 
      # TODO: 
      # try using fnlwgt as a sample_weight in model's fit()
      # try using fnlwgt as a logged feature, instead of being standard scaled
      df = df.drop('fnlwgt', axis=1) 
      
      # create new column for net capital
      df["net_capital"] = df["capital_gain"] - df["capital_loss"]
      
      # create new column for is senior
      df["is_senior"] = (df["age"] >= 65).astype(int)
      
      df['income_is_greater_than_50k'] = (df['income'].str.strip() == '>50K').astype(int)
      df = df.drop('income', axis=1)
      
      # split into X and Y. Remember to do this before scaling, since we dont need to scale Y.    
      X = df.drop('income_is_greater_than_50k', axis=1)
      y = df['income_is_greater_than_50k']
      
      # scale numerical values
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X)
      
      print(X.head()) 
      print(y.head()) 
      
      self.df = df
      self.X_scaled = X_scaled
      self.y = y
      self.next(self.get_train_test_split)
      
    @step
    def get_train_test_split(self):
      """
      Get train test split.
      """
      y = self.y
      X_scaled = self.X_scaled

      X_trainval, X_test, y_trainval, y_test = train_test_split(X_scaled, y, test_size=self.TEST_SPLIT, random_state=42)
      X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)
      
      self.X_train = X_train.tolist()
      self.X_val = X_val.tolist()
      self.X_test = X_test.tolist()
      self.y_train = y_train.tolist()
      self.y_val = y_val.tolist()
      self.y_test = y_test.tolist()  
      
      self.next(self.set_hyperparameter_grid_and_start_training)
    
    @step
    def set_hyperparameter_grid_and_start_training(self):
      self.param_grid = [
        {"C": 0.1, "penalty": "l2", "solver": "liblinear", "fit_intercept": True},
        {"C": 1.0, "penalty": "l2", "solver": "liblinear", "fit_intercept": True},
        {"C": 1.0, "penalty": "l1", "solver": "liblinear", "fit_intercept": False},
        {"C": 10.0, "penalty": "l2", "solver": "lbfgs", "fit_intercept": True},
        {"C": 1e-10, "penalty": "l2", "solver": "lbfgs", "fit_intercept": False},
      ]
      self.next(self.train_model, foreach="param_grid")
    
    @step
    def train_model(self):
      
      from sklearn import linear_model
      from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  
      
      params = self.input
      self.hyperparams = params

      model = linear_model.LogisticRegression(**params)
      model.fit(self.X_train, self.y_train)
      y_pred = model.predict(self.X_val)
      
      acc = accuracy_score(self.y_val, y_pred)
      self.val_accuracy = acc
      self.ps = precision_score(self.y_val, y_pred, zero_division=0)
      self.rs = recall_score(self.y_val, y_pred)
      self.f1 = f1_score(self.y_val, y_pred)
      self.roc = roc_auc_score(self.y_val, y_pred)
      
      metrics = {
        'accuracy': self.val_accuracy,
        'precision_score': self.ps,
        'recall_score': self.rs,
        'f1_score': self.f1,
        'auc_roc': self.roc
      }
      
      exp = Experiment(project_name=os.environ['MY_PROJECT_NAME'],
                    auto_param_logging=False)
      exp.log_parameters(params)
      exp.log_metrics(metrics)  
      exp.end()
      
      self.next(self.join_models)
      
    @step
    def join_models(self, inputs):
        # Pick the best model based on validation accuracy
        best = max(inputs, key=lambda x: x.val_accuracy)
        print(best.input)
        print(best.val_accuracy)
        self.next(self.end)

    @step
    def end(self):
      # all done, just print goodbye
      print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))
      
if __name__ == '__main__':
    LogisticRegressionFlow()
