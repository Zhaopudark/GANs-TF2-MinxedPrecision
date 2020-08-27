class ActivationError(Exception):
    def __init__(self,activation):
        self.err_msg = "Unsupported activation func:"
        self.activation = activation
class ShapeError(Exception):
    def __init__(self,shape):
        self.err_msg = "Unsupported shape:"
        self.shape = shape
class KernelShapeError(Exception):
    def __init__(self,shape):
        self.err_msg = "Unsupported KernelShape:"
        self.shape = shape 
class StridesError(Exception):
    def __init__(self,strides):
        self.err_msg = "Unsupported Strides:"
        self.strides = strides 
class InitializerError(Exception):
    def __init__(self,initializer):
            self.err_msg = "Unsupported initializer:"
            self.initializer = initializer 
class ConvParaError(Exception):
    def __init__(self,parameter):
            self.err_msg = "Unsupported Convolution(or Transpose) Parameter:"
            self.parameter = parameter 