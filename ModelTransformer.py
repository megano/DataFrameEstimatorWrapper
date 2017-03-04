from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_curve
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


class ModelDataframeWrapper(TransformerMixin):
    def __init__(self, model) :
        self.model = model

    def transform(self, X, *_):
        cols = X.columns
        ret = self.model.fit_transform(X)
        ret = pd.DataFrame(ret, columns = cols)
        return ret


    def fit(self, X, y =None):
        self.model.fit(X,y)
        return self

class kfold_classification_model(TransformerMixin):
    def __init__(self, model, nfolds = 5):
        self.model = model
        self.nfolds = nfolds

    def fit(self, X_1, y_1):
        X = np.array(X_1.copy())
        y = np.array(y_1.copy())

        self.X = X
        self.y = y

        kf = StratifiedKFold(n_splits=self.nfolds, shuffle=True)

        kscores = []
        models = []
        for train_index, test_index in kf.split(self.X, self.y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            mod = clone(self.model)
            mod.fit(X_train, y_train)
            kscores.append(mod.score(X_test, y_test))
            models.append(mod)
        self.kscores = kscores
        self.model.fit(X, y);
        return self

    def transform(self, X, *_) :
            return self.model.predict()

    def predict(self, X) :
        return pd.DataFrame(self.model.predict())

    def score(self, X_test, y_test, printit=False) :

        acc = self.model.score(X_test, y_test)

        if(printit == True) :
            y_pred = self.model.predict(X_test)

            isBinary = (set(np.unique(y_test)) == {0,1})
            cm = confusion_matrix(y_test, y_pred)


            #scoring parameters

            precision = precision_score(y_test, y_pred, average=None)
            recall = recall_score(y_test, y_pred,average=None)
            #

            print "\nConfusion matrix "
            print cm
            print "Accuracy ", acc
            print "precision ", precision
            print "recall ", recall

            if(isBinary == True) :
                y_pred_proba = self.model.predict_proba(X_test)
                fpr, tpr,thres = roc_curve(y_test, y_pred_proba[:,1])
                plt.plot(fpr, tpr)
                plt.show()

        return np.mean(self.kscores)

class PredictTransformer(TransformerMixin):
    def __init__(self, model) :
        self.model = model

    def transform(self, X, *_):
        ret = self.model.predict(X)
        ret = pd.DataFrame(ret)
        return ret

    def fit(self, X, y =None):
        return self

class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        val = X.copy()
        try :
            val = val.todense()
        except:
            pass
        return val
