from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os
import numpy as np
import pandas as pd
from comet_ml import Experiment
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.multiclass import type_of_target
import uuid
from io import StringIO

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]
TARGET_VALUE = "occupation"
PARAMETER_GRID = [
    {"C": 0.1, "penalty": "l2", "solver": "lbfgs", "fit_intercept": True},
    {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "fit_intercept": True},
    {"C": 10.0, "penalty": "l2", "solver": "lbfgs", "fit_intercept": False},
    {"C": 1.0, "penalty": "l2", "solver": "newton-cg", "fit_intercept": True},
    {"C": 1e-10, "penalty": "l2", "solver": "newton-cg", "fit_intercept": True},
]

class LogisticRegressionFlow(FlowSpec):

    DATA_FILE = IncludeFile('dataset', help='Census data', is_text=True, default='data/data.csv')
    TEST_SPLIT = Parameter('test_split', help='Proportion for test split', default=0.12)

    @step
    def start(self):
        print("Starting run at", datetime.utcnow())
        self.next(self.load_data)

    @step
    def load_data(self):
        df = pd.read_csv(StringIO(self.DATA_FILE), header=None)
        df.columns = COLUMN_NAMES
        df = df.replace(' ?', np.nan).dropna()
        self.df = df.sample(frac=0.3, random_state=42).reset_index(drop=True)
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        df = self.df.copy()
        self.sample_weights = df['fnlwgt'].values
        df = df.drop(columns='fnlwgt')
        df["net_capital"] = df["capital_gain"] - df["capital_loss"]
        df["is_senior"] = (df["age"] >= 65).astype(int)

        cat_cols = df.select_dtypes(include='object').columns.drop(TARGET_VALUE)
        df = pd.get_dummies(df, columns=cat_cols)
        y = LabelEncoder().fit_transform(df[TARGET_VALUE])
        X = df.drop(columns=TARGET_VALUE)
        X_scaled = StandardScaler().fit_transform(X)

        self.X_scaled, self.y = X_scaled, y
        self.next(self.split_data)

    @step
    def split_data(self):
        X_temp, self.X_test, y_temp, self.y_test, w_temp, self.w_test = train_test_split(
            self.X_scaled, self.y, self.sample_weights, test_size=self.TEST_SPLIT, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val, self.w_train, self.w_val = train_test_split(
            X_temp, y_temp, w_temp, test_size=self.TEST_SPLIT, random_state=42)

        for attr in ["X_train", "X_val", "X_test"]:
            setattr(self, attr, getattr(self, attr).tolist())
        for attr in ["y_train", "y_val", "y_test", "w_train", "w_val", "w_test"]:
            setattr(self, attr, getattr(self, attr).tolist())

        self.param_grid = PARAMETER_GRID
        self.next(self.train_model, foreach='param_grid')

    @step
    def train_model(self):
        params = self.input
        model = linear_model.LogisticRegression(**params, multi_class='multinomial', random_state=42)
        model.fit(self.X_train, self.y_train, sample_weight=self.w_train)
        y_pred = model.predict(self.X_val)

        avg = 'macro' if type_of_target(self.y_val) != 'binary' else 'binary'
        metrics = {
            'accuracy': accuracy_score(self.y_val, y_pred, sample_weight=self.w_val),
            'precision': precision_score(self.y_val, y_pred, average=avg, zero_division=0, sample_weight=self.w_val),
            'recall': recall_score(self.y_val, y_pred, average=avg, sample_weight=self.w_val),
            'f1': f1_score(self.y_val, y_pred, average=avg, sample_weight=self.w_val)
        }

        exp = Experiment(project_name=os.environ['MY_PROJECT_NAME'], auto_param_logging=False)
        exp.log_parameters(params)
        exp.log_metrics(metrics)
        exp.end()

        self.metrics = metrics
        self.model_params = params
        self.next(self.select_best_model)

    @step
    def select_best_model(self, inputs):
        best = max(inputs, key=lambda x: x.metrics['accuracy'])
        self.best_params = best.model_params
        self.X_train = best.X_train + best.X_val
        self.y_train = best.y_train + best.y_val
        self.w_train = best.w_train + best.w_val
        self.X_test = best.X_test
        self.y_test = best.y_test
        self.w_test = best.w_test
        self.next(self.final_eval)

    @step
    def final_eval(self):
        model = linear_model.LogisticRegression(**self.best_params, multi_class='multinomial', random_state=42)
        model.fit(self.X_train, self.y_train, sample_weight=self.w_train)
        y_pred = model.predict(self.X_test)

        avg = 'macro' if type_of_target(self.y_test) != 'binary' else 'binary'
        final_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred, sample_weight=self.w_test),
            'precision': precision_score(self.y_test, y_pred, average=avg, zero_division=0, sample_weight=self.w_test),
            'recall': recall_score(self.y_test, y_pred, average=avg, sample_weight=self.w_test),
            'f1': f1_score(self.y_test, y_pred, average=avg, sample_weight=self.w_test)
        }

        exp = Experiment(project_name=os.environ['MY_PROJECT_NAME'], auto_param_logging=False, 
                         experiment_name=f"best-{uuid.uuid4().hex[:6]}")
        exp.log_parameters(self.best_params)
        exp.log_metrics(final_metrics)
        exp.end()

        print("Final test metrics:", final_metrics)
        self.next(self.end)

    @step
    def end(self):
        print("All done at", datetime.utcnow())

if __name__ == '__main__':
    LogisticRegressionFlow()
