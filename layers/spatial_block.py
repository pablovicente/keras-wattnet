from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Permute, Lambda, Concatenate

from .attention import AttentionBlock

class SpatialBlock(Layer):
    
    def __init__(self, stack, dilation, w_dim, **kwargs):
        """
        Performs Scaled Dot Attention across the temporal layer. 

        Parameters
        ----------
        stack: int
            Stack number
        dilation: int
            Dilation of the prior residual block 
        w_dim: int

        """
        self.stack = stack
        self.dilation = dilation
        self.w_dim = w_dim

        self.attn_block = None
        
        super(SpatialBlock, self).__init__(**kwargs)
        
    def build(self, input_shape):

        with K.name_scope(self.name):
            
            self.attn_block = AttentionBlock(w_dim=self.w_dim, name=f'AttentionBlock_{self.stack}_{self.dilation}')
                    
        super(SpatialBlock, self).build(input_shape)  # done to make sure self.built is set True
        
    def call(self, inputs, mask=None, **kwargs):
    
        # Attention - Slice across temporal dimension
        x_att_list = []
        for k in range(inputs.shape[1]):
                
            x_att = Lambda(lambda x:x[:,k,:], name=f'Slice_{self.stack}_{self.dilation}_{k}')(inputs)
            x_att = self.attn_block(x_att)

            x_att_list.append(x_att)
                    
        output = Concatenate(axis=2)(x_att_list) # N C W
        output = Permute(dims=(2, 1))(output) # N W C

        return output