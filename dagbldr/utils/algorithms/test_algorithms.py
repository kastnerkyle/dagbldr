import numpy as np
from dagbldr.utils import minibatch_kmedians


def test_kmedians():
    random_state = np.random.RandomState(1999)
    Xa = random_state.randn(200, 2)
    Xb = .25 * random_state.randn(200, 2) + np.array((5, 3))
    X = np.vstack((Xa, Xb))
    ind = np.arange(len(X))
    random_state.shuffle(ind)
    X = X[ind]
    M1 = minibatch_kmedians(X, n_iter=1, random_state=random_state)
    M2 = minibatch_kmedians(X, M1, n_iter=1000, random_state=random_state)
