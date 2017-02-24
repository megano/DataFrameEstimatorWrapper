import numpy as np
import math as m

sigmoid = lambda x : 1/(1 + m.exp(-x))
sigmoidVector = np.vectorize(sigmoid)

#dependent : sigmoidVector
def broadcast_sigmoid(arr) :
    return sigmoidVector(arr)

'''General Distance metrics'''

def euclidean_distance(a, b = None) :
    a = np.array(a)
    if not b :
        b = np.zeros(a.shape)
    else :
        b = np.array(b)
    return m.sqrt(np.sum([(x - y) ** 2 for x,y in zip(a, b)]))

# |A|
#dependent : euclidean_distance
def mod(a) :
    return euclidean_distance(a)

#dependent : mod
def cosine_similarity(a, b) :
    a = np.array(a)
    b = np.array(b)
    return a.dot(b)/(mod(a) * mod(b))

#dependent : cosine_similarity
def cosine_distance(a, b) :
    return 1 - cosine_similarity(a, b)

#kfold validation given data and estimator
