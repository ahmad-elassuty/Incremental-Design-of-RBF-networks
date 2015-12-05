import numpy as np

# Neural Network class
class ID_RBF:

  # Constructor
  # Constructs the intial structure of the RBF network
  #   implemented by the paper
  # Inputs:
  #  inputNodes : number of input nodes
  #  outputNodes: number of output nodes
  def __init__(self, inputNodes, outputNodes):
    # input & hidden -> +1 bias node
    self.ni = inputNodes
    self.nh = 0
    self.no = outputNodes

    # Initializing nodes width
    # width vector of the hidden nodes
    self.wh = np.array([1.])

    # hidden matrix
    # row: represents the output of pattern x
    # col: represents the corresponding RBF unit
    self.ah = np.matrix([])

    # output matrix
    # row: represents the output of pattern x
    # col: represents the corresponding output unit
    self.ao = np.matrix([])

    # Initializing weights
    # Centers are empty since
    #   initially no hidden nodes
    # row: represents the corresponding RBF unit
    # col: represents the center value 
    #   with respect to the corresponding input node
    #  *bias: no bias associated with RBF units
    self.centers = np.matrix([])

    # Directly connect the output
    # matrix defines the connections from the input layer to
    # the output layer
    # col: represents the corresponding output unit
    # row: represents the weight value 
    #   with respect to the connected node
    #   *bias: is the first row of the matrix
    self.wo = np.matrix([[1.]*self.no]*(self.ni+1))


    # Final error of iteration i
    # col: for the corresponding output node
    # row: for each input pattern
    self.error = np.matrix([])

    self.RMSE = {}

  def train(self, patterns, targets):
    if self.ni is not patterns.shape[1]:
      print "Specified patterns have different dimensions from network input"
      return

    self.patterns = patterns
    self.targets = targets
    # number of training patterns
    self.np = self.targets.shape[0]

    if self.nh == 0:
      # First time to train
      # No hidden RBF units

      # calculate error vector
      self.ao = np.column_stack(([1.]*self.np, self.patterns)) * self.wo
      self.error = self.calError()

      centersIndexes = np.abs(self.error).argmax(0).A1

      for row in centersIndexes:
        center = self.patterns[row]

        if self.nh == 0:
          # add first RBF unit
          self.centers = center
          self.wo = np.matrix([[1.0]*self.no]*(2))
          self.wh = np.append(self.wh, 1.)
          self.nh += 1
        else:
          self.addUnit(center)
    iter = 1

    self.calRMSE(iter)
    self.calHiddenActivations()

    jacobianMat = self.calJacobianMat()


    print self.error

  def addUnit(self, center):
    self.centers = np.row_stack((self.centers, center))
    self.wo = np.row_stack((self.wo, [1.0]*self.no))
    self.wh = np.append(self.wh, 1.0)
    self.nh += 1

  def calError(self):
    return self.targets - self.ao

  def calRMSE(self, iter):
    self.RMSE[iter] = np.sqrt(1.0/self.np * sum(np.power(self.error, 2)))
    return self.RMSE[iter]

  # def sqrEuclideanNorm(self):

  def calHiddenActivations(self):
    self.ah = np.empty((self.np, self.nh))
    for i, pattern in enumerate(self.patterns):
      numerator = np.sum(np.power(pattern - self.centers, 2), axis=1)
      self.ah[i] = np.exp(-1. * (numerator.T / self.wh[1:]))
    return self.ah

  def calJacobianMat(self):
    # bias weight
    jacobianMat = np.full((self.np, 1), -1.)

    # output weights
    jacobianMat = np.column_stack((jacobianMat, -1. * self.ah))

    # centers

    x = - 1. * np.multiply(self.ah, self.wo[1:])

    temMat = np.empty((self.np,1))
    for i, center in enumerate(self.centers):
      # center represents one hidden unit
      y = self.patterns - center
      c = np.multiply(2. * x[:,i], y) / self.wh[i+1]
      jacobianMat = np.column_stack((jacobianMat, c))

      # cal hidden widths
      y = np.sum(np.power(y, 2), axis= 1)
      z = np.multiply(x[:,i], y) / np.power(self.wh[i+1], 2)

      # concatenate it to tem
      temMat = np.column_stack((temMat, z))

    # concatenate it to output mat
    jacobianMat = np.column_stack((jacobianMat, temMat[:,1:]))

    return jacobianMat

  def calQuasiHessianMat(self):


  # def activate(self, pattern):
  #   return pattern