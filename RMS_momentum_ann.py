import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt
from util import get_normalized_data
from sklearn.utils import shuffle
from

if __name__ == "__main__":
    X, Y = get_normalized_data()
    X, Y = shuffle(X, Y)

    N = len(X)
    Ntrain = 0.9 * N
    Xtrain, Ytrain = X[:Ntrain, :], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:, :], Y[Ntrain:]

    # implmenet cross_validation later
    model = ann_prop_mom()
    model.fit(Xtrain, Ytrain)
    print "Test score:", model.score(Xtest, Ytest)
