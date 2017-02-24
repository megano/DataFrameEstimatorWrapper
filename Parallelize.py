from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from datetime import datetime

class Parallelize(BaseEstimator, TransformerMixin):
    def __init__(self, transformers):
        self.transformers = {x[0]:x[1] for x in transformers}

    def transform(self, X, *_):
        return reduce(lambda x,y : pd.merge(x,y, right_index = True, left_index = True), [self.transformers[y].transform(X) for y in self.transformers])

    def fit(self, X, y=None):
        for x in self.transformers:
            self.transformers[x] = self.transformers[x].fit(X,y)
        return self
