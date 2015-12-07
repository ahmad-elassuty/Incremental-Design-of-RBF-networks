import numpy as np
from id_rbf import ID_RBF


# x = ID_RBF(2,1)
# patterns = np.matrix([[0,0],[0,1],[1,0], [1,1]])
# targets = np.matrix([[0], [1], [1], [0]])
# x.train(patterns, targets)


# print 'DONE'
# # testPatterns = np.matrix([[1,1],[0,0]])
# print x.activate(patterns)



net = ID_RBF(1,1)
x = np.linspace(0, 6, 201)
patterns = np.asmatrix(x.reshape(201,1))
targets = np.asmatrix(np.sin(x).reshape(201,1))

net.train(patterns,targets)

print 'DONE'
# testPatterns = np.matrix([[1,1],[0,0]])
print net.activate(patterns)
