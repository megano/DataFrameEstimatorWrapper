from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold
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
        X = X_1.copy()
        y = y_1.copy()

        self.X = X
        self.y = y

        kf = KFold(n_splits=self.nfolds, shuffle=True)

        kscores = []
        models = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            mod = clone(self.model)
            mod.fit(X_train, y_train)
            kscores.append(mod.score(X_test, y_test))
            models.append(mod)
        self.kscores = kscores
        self.model.fit(X_1, y_1);
        return self

    def transform(self, X, *_) :
            return self.model.predict()

    def predict(self, X) :
        return pd.DataFrame(self.model.predict())

    def score(self, X_test, y_test) :
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)


        #scoring parameters
        acc = self.model.score(X_test, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        #

        print "\nConfusion matrix "
        print cm
        print "Accuracy ", acc
        print "precision ", precision
        print "recall ", recall

        y_pred_proba = self.model.predict_proba(X_test)
        fpr, tpr,thres = roc_curve(y_test, y_pred_proba[:,1])
        plt.plot(fpr, tpr)
        plt.show()

        return acc