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
    self.no = outputNodes

    # initialize nodes width
    # width vector of the hidden nodes
    self.wh = np.matrix([])

    # output vector of the hidden nodes
    self.ao = np.matrix([])

    # initialize weights
    # centers intially empty since
    # initially no hidden nodes
    self.centers = []

    # Directly connect the output
    # matrix defines the connections from the input layer to
    # the output layer
    # col: specifies the corresponding output unit
    # row: specifies the corresponding weight
    #   bias: is the first row of the matrix
    self.wo = np.matrix([[1.0]*self.ao]*(self.ni+1))

  def train(self, patterns):
    if self.ni is not patterns.shape[1]:
      assert "specified patterns have different dimensions as input"


    return "trained"


  def addUnit():
    pass

  def activate(pattern):
    return pattern