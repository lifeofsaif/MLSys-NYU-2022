from datetime import datetime
from matplotlib import pyplot as plt
from metaflow import FlowSpec, step, IncludeFile , current, card
from metaflow.cards import Markdown, Image
import os
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score



'''
USERNAME=lifeofsaif python homework_5.py run 
'''

class LogisticRegressionFlow(FlowSpec):
    
    DATA_FILE = IncludeFile('dataset', help='Census data', is_text=True, default='data/data.csv')
    
    @step
    def start(self):
        print("Starting run at", datetime.utcnow())
        self.next(self.load_data)
        
    @step
    def load_data(self):
        print("Loading data")
        
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
        ]        
        
        df = pd.read_csv(StringIO(self.DATA_FILE))
        df.columns = column_names        
        
        df = df.replace(' ?', np.nan).dropna()
        df = df.drop(columns='fnlwgt')
        
        cat_cols = df.select_dtypes(include='object').columns
        df = pd.get_dummies(df, columns=cat_cols)
        
        np.random.seed(42)  # for reproducibility
        df["loan_approval"] = np.random.choice([0, 1], size=len(df))
        
        X = df.drop(columns='loan_approval')
        X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns) 
        
        y = df[['loan_approval']]
        
        self.X_scaled = X_scaled
        self.y = y
        self.df = df
        self.next(self.split_data)        
        
    @step
    def split_data(self):
        print("Splitting data")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42)
            
        self.next(self.train_model_and_evaluate)        
        
    def printAccuracy(self, comment, column_name):
        mask = self.X_test[column_name] > 0
        ac = accuracy_score(self.y_pred[mask], self.y_test[mask])
        print(comment, ac)
        return ac
    
    @card(type="blank")
    @step
    def train_model_and_evaluate(self):
        print("training model")
        params = {"C": 0.1, "penalty": "l2", "solver": "lbfgs", "fit_intercept": True}
        
        model = linear_model.LogisticRegression(**params, random_state=42)
        model.fit(self.X_train, self.y_train)
    
        y_pred = model.predict(self.X_test)
        y_pred = pd.DataFrame(y_pred, columns=self.y_test.columns)
        y_pred = y_pred.reset_index(drop=True)
        self.y_pred = y_pred
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Overall Accuracy
        # since our y data was randomly generated, we can get close to random accuracy 0.5, or very slightly over
        print("accuracy: ", accuracy)

        current.card.append(
            Markdown("# Accuracies by Slice")
        )
        
        # gender split accuracy
        maleAccuracy = self.printAccuracy("male accuracy: ", 'sex_ Male')
        femaleAccuracy = self.printAccuracy("female accuracy: ", 'sex_ Female')
    
        # income split accuracy    
        incomeGreaterThan50kAccuracy = self.printAccuracy("income > 50k accuracy: ", 'income_ >50K')
        incomeLessThan50kAccuracy = self.printAccuracy("income <= 50k accuracy: ", 'income_ <=50K')
        
        # countries accuracy
        # usa and mexico have closer to random accuracy because there are more rows
        usaAccuracy = self.printAccuracy("United-States accuracy: ", 'native_country_ United-States')
        mexicoAccuracy = self.printAccuracy("Mexico accuracy: ", 'native_country_ Mexico')
        
        # these ones have less random accuracy since they dont have so many rows
        japanAccuracy = self.printAccuracy("Japan accuracy: ", 'native_country_ Japan')
        thailandAccuracy = self.printAccuracy("Thailand accuracy: ", 'native_country_ Thailand')
        
        categories = ['male', 'female', '$ > 50k', '$ < 50k', 'usa', 'mexico', 'japan', 'thailand']
        
        # Example data
        values = [maleAccuracy, femaleAccuracy, 
                  incomeGreaterThan50kAccuracy, incomeLessThan50kAccuracy,
                  usaAccuracy, mexicoAccuracy, japanAccuracy, thailandAccuracy
                    ]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Plot bar chart
        ax.bar(categories, values)

        # Set titles and labels
        ax.set_title('Accuracies by Slice')
        ax.set_xlabel('Category')
        ax.set_ylabel('Accuracy')

        # Add gridlines
        ax.grid(axis='y')
        
        current.card.append(
            Image.from_matplotlib(
                fig
            )
        )
        
        self.model = model
        self.next(self.end)
        
    @step
    def end(self):
        print("All done at", datetime.utcnow())
        
if __name__ == '__main__':
    LogisticRegressionFlow()