import numpy as np
from id_rbf import ID_RBF

# xor
# net = ID_RBF(2,1)
# patterns = np.matrix([[0,0],[0,1],[1,0], [1,1]])
# targets = np.matrix([[0], [1], [1], [0]])
# net.train(patterns, targets)

# print 'DONE'
# print net.activate(patterns)

# 3 parity xor
# net = ID_RBF(3,1)
# patterns = np.matrix([[0,0,0],[0, 0,1],[0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
# targets = np.matrix([[0], [1], [1], [0], [1], [0], [0], [1]])
# net.train(patterns, targets)

# print 'DONE'
# print net.activate(patterns)


# ---1 Sin approximation (sin(x))
# net = ID_RBF(1,1)
# x = np.linspace(0, 6, 201)
# patterns = np.asmatrix(x.reshape(201,1))
# targets = np.asmatrix(np.sin(x).reshape(201,1))
# net.train(patterns,targets)

# print 'DONE'
# x_test = np.linspace(2, 20, 201)
# patterns_test = np.asmatrix(x_test.reshape(201,1))
# net.plot(patterns_test)

# ---2
net = ID_RBF(1,1)
x = np.linspace(0, 10, 3000)
y = 0.8 * np.exp(-0.2 * x) * np.sin(10* x)

patterns = np.asmatrix(x.reshape(3000,1))
targets = np.asmatrix(y.reshape(3000,1))

net.train(patterns, targets)

print 'DONE'
x_test = np.linspace(7, 10, 150)
patterns_test = np.asmatrix(x_test.reshape(150,1))
net.plot(patterns_test)