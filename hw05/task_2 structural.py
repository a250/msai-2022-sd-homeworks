# Implementation of Dependency Injection pattern as a Structural pattern


import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

class IrisData():
    def __init__(self, portion = 0.8):
        
        self.dataset = load_iris()
        X, y = self.dataset['data'], self.dataset['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size = portion)

    def get_train(self):
        return self.X_train, self.y_train
    
    def get_test(self):
        return self.X_test, self.y_test
    
    
class LogisticPrediction():
    def __init__(self):
        self.model = LogisticRegression(max_iter = 500)
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

    
class LinearPrediction():
    def __init__(self):
        self.model = LinearRegression()
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        pred = np.round(self.model.predict(X)).astype(int)
        return pred

    
class Ensemble():
    
    def __init__(self):
        self.dataset = IrisData()
        self.Logist = LogisticPrediction()
        self.Linear = LinearPrediction()
    
    def get_test(self):
        return self.dataset.get_test()
    
    def averaging(self, predictions):
        y = np.zeros(predictions[0].shape[0])
        for p in predictions:
            y += p
        res = np.round(y / len(predictions))
        return np.abs(res).astype(int)
    
    def ensemblePrediction(self, X):
        pred = []
        X_train, y_train = self.dataset.get_train()  # getting data
        self.Logist.fit(X_train, y_train)            # predict by LogisticRegression model
        self.Linear.fit(X_train, y_train)            # predict by LinearRegression model
        pred.append(self.Logist.predict(X))
        pred.append(self.Linear.predict(X))          
        
        res = self.averaging(pred)                   # averaging prediction
        
        return res
    

if __name__ == "__main__":
    facade = Ensemble()

    X_test, y_test = facade.get_test()

    y_pred = facade.ensemblePrediction(X_test)

    print(f'Ensemble accuracy = {accuracy_score(y_test, y_pred)}')
    