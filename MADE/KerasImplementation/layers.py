import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import standard_ops
import numpy as np


# implements a masked version of the keras Dense layer to be used as a hidden
#   or an output layer in MADE
class MaskedDense(Layer):
    # these are parameters that can be passed in to the creation of a new instance, with their default values
    def __init__(self, units,
                 mask,
                 activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(MaskedDense, self).__init__(**kwargs)
        # each instance of the class will have all of these attributes
        
        # self.first_layer = first_layer
        self.mask = mask
        self.activation = activations.get(activation)
        self.units = units
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape, ):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1] # final elt. of input shape--disregard batch

        # note here how the weights matrix dimensions are equal to those of the
        #   mask
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=initializers.get('glorot_uniform'),
                                      name='kernel')

        # bias is a vector, hence the shape argument here
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=initializers.get('zeros'),
                                    name='bias')

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        super(MaskedDense, self).build(input_shape)

    # this function implements the layer's computations
    def call(self, inputs):
        """if self.first_layer:
            self.first_input = inputs
        else: 
            # get first input from layer inputs"""
        hadamard_product = tf.multiply(self.mask, self.kernel)
        input = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(input)
        if rank > 2:
            dot_product = standard_ops.tensordot(input, hadamard_product, axes=[[rank - 1], [0]])
            shape = input.get_shape().as_list()
            output_shape = shape[:-1] + [self.units]
            dot_product.set_shape(output_shape)
        else: 
            if not self._mixed_precision_policy.should_cast_variables:
                input = math_ops.cast(inputs, self.dtype)
            dot_product = gen_math_ops.mat_mul(input, hadamard_product)
        # multiply the inputs by the hadamard (elementwise) product of the
        #   mask matrix and the weights matrix
        output = K.bias_add(dot_product, self.bias, data_format='channels_last')
        # pass the result of the layer through the activ. function
        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)