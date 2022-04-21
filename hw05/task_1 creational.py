# Implementation of Dependency Injection as a Creational pattern


import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression



class DataPreprocessor(ABC):
    
    @abstractmethod
    def get_train(portion_of_xtrain: float = 1) -> Tuple[List[float], List[float]]:
        pass
    
    @abstractmethod
    def get_test() -> Tuple[List[float], List[float]]:
        pass

    
class IrisDataPreprocessor(DataPreprocessor):
    def __init__(self, shuffle = True, share = 0.8):
        
        self.dataset = load_iris()
        self.X_orig = self.dataset['data']
        self.y_orig = self.dataset['target']
        
        self.X = self.X_orig
        self.y = self.y_orig
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None        
        
        self.share = share
        self.n_ind = len(self.X)
        self.n_share = round(len(self.X) * self.share)
        
        self.shuffle = shuffle
        
        self.preprocess()
        
    def shuffler(self):
        all_ind = np.random.permutation(range(self.n_ind))
        self.X = self.X_orig[all_ind,:]
        self.y = self.y_orig[all_ind]
    
    def do_split(self):
        self.X_train = self.X[:self.n_share,:]
        self.y_train = self.y[:self.n_share]
        
        self.X_test = self.X[self.n_share:,:]
        self.y_test = self.y[self.n_share:]

    def preprocess(self):
        if self.shuffle:
            self.shuffler()
            
        self.do_split()
        
    def get_train(self, portion = 1):
        num = int(self.X_train.shape[0] * portion)
        
        return self.X_train[:num, :], self.y_train[:num]
    
    def get_test(self):
        return self.X_test, self.y_test

    
class Prediction():
    
    def __init__(self, data_supplier: DataPreprocessor, classifier):
        
        self.get_train = data_supplier.get_train
        self.get_test = data_supplier.get_test
        
        self.model = classifier
    
    def training(self, portion_of_xtrain = 1):
        X, y = self.get_train(portion = portion_of_xtrain)
        self.model.fit(X, y)

    def train_and_get_train_score(self, portion_of_xtrain = 1):
        X, y = self.get_train(portion = portion_of_xtrain)
        self.model.fit(X, y)        
        return self.model.score(X, y)
    
    def get_test_score(self):
        X, y = self.get_test()
        return self.model.score(X, y)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    

if __name__ == "__main__":
    
    model = Prediction(data_supplier = IrisDataPreprocessor(shuffle = True, share=0.7), 
                       classifier = LinearRegression())
    #                   classifier = LogisticRegression())


    for df_share in range(10, 101, 10):
        print(f'For {df_share}% of x_train:')
        print(f'...training score = {model.train_and_get_train_score(portion_of_xtrain=df_share/100):.3f}')
        print(f'....testing score = {model.get_test_score():.3f}')
        print('\n')