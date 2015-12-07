import numpy as np
from numpy.linalg import inv
np.set_printoptions(precision=15)
import time
# General Methods
def forward(np, nh, patterns, targets, centers, wh, wo):
  ah = calHiddenActivations(np, nh, patterns, centers, wh)
  ao = calOutputActivations(np, ah, wo)
  error = calError(targets, ao)
  return (ah, ao, error)

def calError(targets, ao):
  return targets - ao

def calRMSE(nPatterns, error):
  return np.sqrt((1.0/nPatterns) * sum(np.power(error, 2)))[0,0]

def calHiddenActivations(nPatterns, nh, patterns, centers, wh):
  ah = np.empty((nPatterns, nh))
  for i, pattern in enumerate(patterns):
    numerator = np.sum(np.asmatrix(np.power(pattern - centers, 2)), axis=1)
    ah[i] = np.exp(-1. * (numerator.T / wh))
  return ah

def calOutputActivations(nPatterns, ah, wo):
  return np.column_stack(([1.]*nPatterns, ah)) * wo

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

    target_Error_1 = 0.2
    target_Error_2 = 2.5e-5
    combinationCoef = 10.0

    while True:
      if self.nh == 0:
        # Add first RBF unit
        centerIndex = np.abs(self.targets).argmax(0).A1[0]
        self.centers = self.patterns[centerIndex]
        self.wo = np.matrix([[1.0]*self.no]*(2))
        self.wh = np.append(self.wh, 1.0)
        self.nh += 1
      else:
        centerIndex = np.abs(self.error).argmax(0).A1[0]
        center = self.patterns[centerIndex]
        self.addUnit(center)

      print "RBF units: ", self.nh

      iter = 1
      self.RMSE = {}
      self.__forward()
      self.__calRMSE(iter)

      while True:
        iter += 1

        jacobianMat = self.calJacobianMat()
        quasiHessianMat, gradientVec = self.calQuashiAndGradient(jacobianMat)

        # virtual update
        wo, centers, wh = self.update(quasiHessianMat, gradientVec, combinationCoef, updateNetwork=False)
        ah, ao, error = forward(self.np, self.nh, patterns, targets, centers, wh, wo)
        RMSE = calRMSE(self.np, error)

        # self.update(quasiHessianMat, gradientVec, combinationCoef)
        # self.__forward()
        # RMSE = self.__calRMSE(iter)

        # print RMSE
        # time.sleep(5)
        # print self.RMSE[iter] >= self.RMSE[iter-1]

        print "prev Itr: ", self.RMSE[iter-1]
        print "out RMSE: ", RMSE
        print "Cond: ", RMSE > self.RMSE[iter-1]

        prevRMSE = self.RMSE[iter-1]

        # m = 1
        while RMSE >= prevRMSE:
          # if m > 5:
          #   break
          # m += 1

          combinationCoef *= 10

          if combinationCoef >= np.finfo(np.float64).max:
            combinationCoef = 1./np.finfo(np.float64).max

          print "combinationCoef: ", combinationCoef
          wo, centers, wh = self.update(quasiHessianMat, gradientVec, combinationCoef, updateNetwork=False)
          ah, ao, error = forward(self.np, self.nh, patterns, targets, centers, wh, wo)
          RMSE = calRMSE(self.np, error)
          
          print "inner RMSE: ", RMSE


          # if tem == RMSE:
          #   m += 1
          #   if m >= 10:
          #     coef -= 1.
          # else:
          #   tem = RMSE
          #   coef = 10.
          #   m = 1

          # RMSE_1 = calRMSE(self.np, error)
          
          # print "inner RMSE 1: ", RMSE_1

          # if RMSE_1 > RMSE and not (combinationCoef <= 9.88131291682e-323):
          #   combinationCoef /= 100
          #   print "combinationCoef 1: ", combinationCoef
          #   wo, centers, wh = self.update(quasiHessianMat, gradientVec, combinationCoef, updateNetwork=False)
          #   ah, ao, error = forward(self.np, self.nh, patterns, targets, centers, wh, wo)
          #   RMSE = calRMSE(self.np, error)
          # else:
          #   RMSE = RMSE_1

          
        
        self.wo = wo
        self.centers = centers
        self.wh[1:] = wh
        self.ah = ah
        self.ao = ao
        self.error = error
        self.__calRMSE(iter)

        # self.networkParams()
        if self.RMSE[iter] < prevRMSE:
          combinationCoef /= 10.
        else:
          combinationCoef = 0.0006


        if self.RMSE[iter] <= target_Error_1:
          break

      if self.RMSE[iter] <= target_Error_2:
        break
    
      print self.RMSE[iter]
    #   print self.error
    # print self.wo
    # print self.centers
    # print self.wh
    # print self.ah
    # print self.ao
    # print self.error
    print self.RMSE[len(self.RMSE)]
    print self.nh

  def networkParams(self):
    print "wo : ", self.wo
    print "centers : ", self.centers
    print "wh : ", self.wh
    print "ah : ", self.ah
    print "ao : ", self.ao
    print "error : ",  self.error
    time.sleep(5)

  def update(self, quasiHessianMat, gradientVec, combinationCoef, updateNetwork=True):
    result = np.row_stack((self.wo, self.centers.flatten().T, self.wh[1:][np.newaxis].T))\
      - inv(quasiHessianMat + combinationCoef * np.asmatrix(np.identity(quasiHessianMat.shape[0]))) * gradientVec
    
    outputConnections = self.wo.shape[0]

    wo = result[0:outputConnections]
    centers = np.asmatrix(result[outputConnections: self.nh * self.ni + outputConnections].reshape(self.nh, self.ni))
    wh = np.asarray(result[self.nh * self.ni + outputConnections:].T).flatten()

    if not updateNetwork:
      return (wo, centers, wh)
    
    self.wo = wo
    self.centers = centers
    self.wh[1:] = wh

  def addUnit(self, center):
    self.centers = np.row_stack((self.centers, center))
    self.wo = np.row_stack((self.wo, 1.))
    self.wh = np.append(self.wh, 1.)
    self.nh += 1

  def __calError(self):
    self.error = self.targets - self.ao
    return self.error

  def __calRMSE(self, iter):
    self.RMSE[iter] = np.sqrt((1.0/self.np) * sum(np.power(self.error, 2)))[0,0]
    return self.RMSE[iter]

  def __forward(self):
    self.__calHiddenActivations()
    self.__calOutputActivations()
    return self.__calError()

  def __calHiddenActivations(self):
    self.ah = np.empty((self.np, self.nh))
    for i, pattern in enumerate(self.patterns):
      numerator = np.sum(np.asmatrix(np.power(pattern - self.centers, 2)), axis=1)
      self.ah[i] = np.exp(-1. * (numerator.T / self.wh[1:]))
    return self.ah

  def __calOutputActivations(self):
    self.ao = np.column_stack(([1.]*self.np, self.ah)) * self.wo
    return self.ao

  def calJacobianMat(self):
    # bias weight
    jacobianMat = np.full((self.np, 1), -1.)

    # output weights
    jacobianMat = np.column_stack((jacobianMat, -1. * self.ah))
    
    # centers
    x = - 1. * np.multiply(self.ah, self.wo[1:].T)

    temMat = np.empty((self.np,1))
    for i, center in enumerate(self.centers):
      # center represents one hidden unit
      y = self.patterns - center
      c = (np.multiply(2. * x[:,i], y) / self.wh[i+1])
      jacobianMat = np.column_stack((jacobianMat, c))

      # cal hidden widths
      y = np.sum(np.power(y, 2), axis= 1)
      z = np.multiply(x[:,i], y) / np.power(self.wh[i+1], 2)

      # concatenate it to tem
      temMat = np.column_stack((temMat, z))

    # concatenate it to output mat
    jacobianMat = np.column_stack((jacobianMat, temMat[:,1:]))

    return jacobianMat

  def calQuashiAndGradient(self, jacobianMat):
    dim = jacobianMat.shape[1]
    quasiHessianMat = np.asmatrix(np.zeros((dim, dim)))
    gradientVec = np.asmatrix(np.zeros((dim, 1)))
    for index, jacobianVec in enumerate(jacobianMat):
      quasiHessianMat += self.calSubQuasiHessianMat(jacobianVec)
      gradientVec += self.calSubGradientVec(jacobianVec, index)

    return (quasiHessianMat, gradientVec)

  def calSubQuasiHessianMat(self, jacobianVec):
    return jacobianVec.T * jacobianVec

  def calSubGradientVec(self, jacobianVec, index):
    return jacobianVec.T * self.error[index,0]

  def activate(self, patterns):
    ah = calHiddenActivations(patterns.shape[0], self.nh, patterns, self.centers, self.wh[1:])
    ao = calOutputActivations(patterns.shape[0], ah, self.wo)
    return ao