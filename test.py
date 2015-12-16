import numpy as np
from id_rbf import ID_RBF

# xor
# combinationCoef = 100.0
# without abs widths

# net = ID_RBF(2,1)
# patterns = np.matrix([[0,0],[0,1],[1,0], [1,1]])
# targets = np.matrix([[0], [1], [1], [0]])
# net.train(patterns, targets)

# print 'DONE'
# print net.activate(patterns)

# 3 parity xor
# combinationCoef = .001 -> long training time -> 4 rbf units
# combinationCoef = .01 -> fair training time -> 5 rbf units
# combinationCoef = 10. -> fair training time -> 4 rbf units
# without abs widths

# net = ID_RBF(3,1)
# patterns = np.matrix([[0, 0, 0],
#                       [0, 0, 1],
#                       [0, 1, 0],
#                       [0, 1, 1],
#                       [1, 0, 0],
#                       [1, 0, 1],
#                       [1, 1, 0],
#                       [1, 1, 1]])
# targets = np.matrix([[0], [1], [1], [0], [1], [0], [0], [1]])
# net.train(patterns, targets)

# print 'DONE'
# print net.activate(patterns)


# ---1 Sin approximation (sin(x))
net = ID_RBF(1,1)
x = np.linspace(0, 6, 400)
patterns = np.asmatrix(x.reshape(400,1))
targets = np.asmatrix(np.sin(x).reshape(400,1))
net.train(patterns,targets)

print 'DONE'
x_test = np.linspace(2, 20, 201)
patterns_test = np.asmatrix(x_test.reshape(201,1))
net.plot(patterns_test)

# ---2
# net = ID_RBF(1,1)
# x = np.linspace(0, 10, 3000)
# y = 0.8 * np.exp(-0.2 * x) * np.sin(10* x)

# patterns = np.asmatrix(x.reshape(3000,1))
# targets = np.asmatrix(y.reshape(3000,1))

# net.train(patterns, targets)

# print 'DONE'
# x_test = np.linspace(7, 10, 150)
# patterns_test = np.asmatrix(x_test.reshape(150,1))
# net.plot(patterns_test)

# --3
# net = ID_RBF(2,1)

# x_1 = np.random.uniform(0,1, 50)
# x_1 = np.concatenate((x_1, np.random.uniform(0,1,50)))
# x_1 = np.concatenate((x_1, np.random.uniform(2,3,50)))
# x_1 = np.concatenate((x_1, np.random.uniform(2,3,50)))


# x_2 = np.random.uniform(0, 1, 50)
# x_2 = np.concatenate((x_2, np.random.uniform(3, 4,50)))
# x_2 = np.concatenate((x_2, np.random.uniform(0, 1,50)))
# x_2 = np.concatenate((x_2, np.random.uniform(3, 4,50)))

# y = np.array([0.0]*50)
# y = np.concatenate((y, np.array([1.0]*50)))
# y = np.concatenate((y, np.array([2.0]*50)))
# y = np.concatenate((y, np.array([3.0]*50)))

# patterns = np.asmatrix(np.column_stack((x_1, x_2)))
# targets = np.asmatrix(y.reshape(200,1))

# net.train(patterns, targets)

# x_1_test = np.random.uniform(0, 1, 10)
# x_2_test = np.random.uniform(0, 2, 10)

# x_1_test = np.concatenate((x_1_test ,np.random.uniform(0, 1, 10)))
# x_2_test = np.concatenate((x_2_test ,np.random.uniform(4, 5, 10)))

# x_1_test = np.concatenate((x_1_test ,np.random.uniform(2, 3, 10)))
# x_2_test = np.concatenate((x_2_test ,np.random.uniform(1, 2, 10)))

# x_1_test = np.concatenate((x_1_test ,np.random.uniform(3, 4, 10)))
# x_2_test = np.concatenate((x_2_test ,np.random.uniform(3, 4, 10)))

# patterns_test = np.asmatrix(np.column_stack((x_1_test, x_2_test)))

# print 'DONE'
# print net.activate(patterns_test)