from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Conv1D, Multiply, LayerNormalization

class ResidualBlock(Layer):
    
    def __init__(self, dilation, filters, kernel_size, groups, **kwargs):
        """
        Residual block similar to WaveNet.

        Reference: https://arxiv.org/pdf/1609.03499.pdf

        Parameters
        ----------
        dilation: int
            Dilation of the prior residual block 
        filters: int
            Number of output channels
        kernel_size: int
            Length of the 1 dimensional filter
        groups: int
            Number of groups for the grouped convolution
        """
        self.dilation = dilation
        self.filters = filters
        self.kernel_size = kernel_size
        self.groups = groups

        self.tanh_conv1d = None
        self.sigmoid_conv1d = None
        self.same_conv1d = None
        
        super(ResidualBlock, self).__init__(**kwargs)
        
    def build(self, input_shape):

        with K.name_scope(self.name):
            
            self.tanh_conv1d = Conv1D(filters=self.filters, 
                                      kernel_size=self.kernel_size, 
                                      dilation_rate=self.dilation, 
                                      groups=self.groups, 
                                      padding='causal',
                                      activation='tanh')
            
            self.sigmoid_conv1d = Conv1D(filters=self.filters,
                                         kernel_size=self.kernel_size,
                                         dilation_rate=self.dilation,
                                         groups=self.groups,  
                                         padding='causal',
                                         activation='sigmoid')          
                
            self.same_conv1d = Conv1D(filters=self.filters, kernel_size=1, groups=self.filters, padding='same')
        
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs):
        
        tanh_conv = self.tanh_conv1d(inputs)
        sigmoid_conv = self.sigmoid_conv1d(inputs)

        x_out = Multiply()([tanh_conv, sigmoid_conv])
        x_out = self.same_conv1d(x_out)

        return x_out