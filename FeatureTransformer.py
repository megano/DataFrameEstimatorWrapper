from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class imputing_transformer(TransformerMixin):
    def __init__(self, transformer, empty_values = [float('nan'), np.NaN, None]):
        self.transformer = transformer
        self.empty_values = empty_values

    def transform(self, X, *_):
        # impute = np.vectorize(lambda x :  1 if x in self.empty_values else 0)
        # replcae_b4 = np.vectorize(lambda x : np.NaN  if x in self.empty_values else x)

        imputed = X.applymap(lambda x :  1 if x in self.empty_values else 0)
        imputed.columns = [col+"_imp" for col in X.columns]
        X = X.applymap(lambda x : np.NaN  if x in self.empty_values else x)
        date_frm = pd.DataFrame(self.transformer.fit(X).transform(X))
        date_frm.columns = X.columns
        return pd.merge(date_frm, imputed, right_index= True, left_index = True)

    def fit(self, *_):
        return self

class HotEncoder(TransformerMixin):

    def transform(self, X, *_):
        return pd.get_dummies(X, drop_first=False)

    def fit(self, *_):
        return self
