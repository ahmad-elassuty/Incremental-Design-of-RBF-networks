import numpy as np
from id_rbf import ID_RBF


x = ID_RBF(2,1)
patterns = np.matrix([[0,0],[0,1],[1,0], [1,1]])
targets = np.matrix([[0],[1], [1], [0]])
x.train(patterns, targets)