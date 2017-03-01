from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from datetime import datetime

class Parallelize(BaseEstimator, TransformerMixin):
    def __init__(self, transformers):
        self.transformers = {x[0]:x[1] for x in transformers}

    def transform(self, X, *_):
        ret =  reduce(lambda x,y : pd.merge(x,y, right_index = True, left_index = True), [pd.DataFrame(self.transformers[y].transform(X)) for y in self.transformers])
        ret.reset_index(inplace = True, drop = True)
        return ret

    def fit(self, X, y=None):
        for x in self.transformers:
            self.transformers[x] = self.transformers[x].fit(X,y)
        return self
