import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile, upper_quantile):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lower_quantile_ = np.quantile(X, self.lower_quantile)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile)
        return self


    def transform(self, X):
        X = np.clip(X, a_min=self.lower_quantile_, a_max= self.upper_quantile_)
        return X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
