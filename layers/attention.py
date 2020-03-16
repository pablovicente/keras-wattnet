from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense, Flatten, Lambda, Reshape

class ScaledDotAttention(Layer):
    
    def __init__(self, **kwargs):
        """
        Scale Dot Attention

        Reference: https://arxiv.org/pdf/1706.03762.pdf

        https://github.com/Lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py
        """
        super(ScaledDotAttention, self).__init__(**kwargs)
        
    def compute_mask(self, inputs, mask=None):

        if isinstance(mask, list):
            mask = mask[0]

        return mask

    def call(self, inputs, mask=None, **kwargs):
            
        key, query, value = inputs
        
        if isinstance(mask, list):
            mask = mask[1]
        
        feature_dim = K.shape(query)[-1]
    
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))

        if mask is not None:
            e *= K.cast(K.expand_dims(mask, axis=-2), K.floatx())

        a = e / (K.sum(e, axis=-1, keepdims=True) + K.epsilon())
        v = K.batch_dot(a, value)

        return v    
    
    
class AttentionBlock(Layer):
    
    def __init__(self, w_dim, **kwargs):
        """
        Prepares the output of the Residual block and performs Scaled 
        Dot Attention across the temporal dimension. 
        
        Parameters
        ----------
        w_dim: int
            Size of the feature dimention number

        """
        self.w_dim = w_dim

        super(AttentionBlock, self).__init__(**kwargs)
    
    
    def build(self, input_shape):

        # Dimensions
        bs, seq_len = input_shape
        ltransf_dim = self.w_dim * 1

        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            
            self.linear_key = Dense(units=ltransf_dim)            
            self.linear_query = Dense(units=ltransf_dim)            
            self.linear_value = Dense(units=ltransf_dim)
                    
        super(AttentionBlock, self).build(input_shape)  # done to make sure self.built is set True
        
    def call(self, inputs, mask=None, **kwargs):
    
        # Dimensions
        bs, seq_len = inputs.shape

        inputs = Flatten()(inputs)

        # Inputs -> Key, query & value 
        key = self.linear_key(inputs)
        key = Reshape((self.w_dim, 1))(key)  # `N, W, key_size`

        query = self.linear_query(inputs)
        query = Reshape((self.w_dim, 1))(query)  # `N, W, key_size`

        value = self.linear_value(inputs)
        value = Reshape((self.w_dim, 1))(value)  # `N, W, value_size`

        output = ScaledDotAttention()([key, query, value])

        return output