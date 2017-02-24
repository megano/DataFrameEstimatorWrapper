import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(estimator, X, y, label=None):
    ''' Plot learning curve with varying training sizes'''
    scores = list()
    train_sizes = np.linspace(10,100,10).astype(int)
    for train_size in train_sizes:
        cv_shuffle = cross_validation.ShuffleSplit(train_size=train_size,
                        test_size=200, n=len(y), random_state=0)
        test_error = cross_validation.cross_val_score(estimator, X, y, cv=cv_shuffle)
        scores.append(test_error)

    plt.plot(train_sizes, np.mean(scores, axis=1), label=label or estimator.__class__.__name__)
    plt.ylim(0,1)
    plt.title('Learning Curve')
    plt.ylabel('Explained variance on test set (R^2)')
    plt.xlabel('Training test size')
    plt.legend(loc='best')
    plt.show()

#scatter
#QQ plot
#kde
#boxplot
#influence_plot

#roc

#partial dependence plot
