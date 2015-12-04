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

    # Initializing nodes width
    # width vector of the hidden nodes
    self.wh = np.matrix([])

    # output vector of the hidden nodes
    self.ao = np.matrix([])

    # Initializing weights
    # Centers are empty since
    #   initially no hidden nodes
    # col: represents the corresponding RBF unit
    # row: represents the center value 
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
    self.wo = np.matrix([[1.0]*self.ao]*(self.ni+1))

  def train(self, patterns):
    if self.ni is not patterns.shape[1]:
      assert "specified patterns have different dimensions as input"
    return "trained"


  def addUnit():
    pass

  def activate(pattern):
    return pattern