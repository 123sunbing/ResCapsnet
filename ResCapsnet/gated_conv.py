from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,Add
from keras.layers import Multiply

#Defining CRAM block
def block(x, n_filters=64, pool_size=(2, 2), dropout_rate=0.2):
    """Apply two gated convolutions followed by a max-pooling operation.

    Batch normalization and dropout are applied for regularization.

    Args:
        x (tensor): Input tensor to transform.
        n_filters (int): Number of filters for each gated convolution.
        pool_size (int or tuple): Pool size of max-pooling operation.
        dropout_rate (float): Fraction of units to drop.

    Returns:
        A Keras tensor of the resulting output.
    """
    x1 = GatedConv(n_filters, padding='same')(x)
    x2 = BatchNormalization(axis=-1)(x1)
    x3 = Activation('relu')(x2)
    x4 = Dropout(rate=dropout_rate)(x3)
    x5 = GatedConv(n_filters, padding='same')(x4)
    x6 = BatchNormalization(axis=-1)(x5)
    x7 = Dropout(rate=dropout_rate)(x6)
    x8 = Add()([x,x7])
    x = MaxPooling2D(pool_size=pool_size)(x8)
    
    return Activation('relu')(x)


class GatedConv(Conv2D):
   
    def __init__(self, n_filters=64, kernel_size=(3, 3), **kwargs):
        super(GatedConv, self).__init__(filters=n_filters*2,
                                        kernel_size=kernel_size,
                                        **kwargs)

        self.n_filters = n_filters

    def call(self, inputs):
        """Apply gated convolution."""
        output = super(GatedConv, self).call(inputs)

        n_filters = self.n_filters
        linear = Activation('linear')(output[:, :, :, :n_filters])
        sigmoid = Activation('sigmoid')(output[:, :, :, n_filters:])

        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output."""
        output_shape = super(GatedConv, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.n_filters,)

    def get_config(self):
        """Return the config of the layer."""
        config = super(GatedConv, self).get_config()
        config['n_filters'] = self.n_filters
        del config['filters']
        return config
