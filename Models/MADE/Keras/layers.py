import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import backend
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
    # these are parameters that can be passed in to the creation of a new
    #   instance, with their default values
    def __init__(self, units,
                 mask,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
 
        # each instance of the class will have all of these attributes
        self.mask_tensor = mask
        self.activation = activations.get(activation)
        self.units = units
        self.use_bias = use_bias
        self.input_spec = InputSpec(min_ndim=2)
        super(MaskedDense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        # note here how the weights matrix dimensions are equal to those of the
        #   mask
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=initializers.get('glorot_uniform'),
                                      dtype=self.dtype,
                                      trainable=True,
                                      name='kernel')

        # bias is a vector, hence the shape argument here
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=initializers.get('zeros'),
                                        dtype=self.dtype,
                                        trainable=True,
                                        name='bias')
        else:
            self.bias = None
        self.mask = self.add_weight(shape=self.mask_tensor.shape,
                                    trainable=False,
                                    dtype=self.dtype,
                                    name='mask')
        self.mask.assign(value=self.mask_tensor)
        super(MaskedDense, self).build(input_shape)
        self.built = True

    # this function implements the layer's computations
    def call(self, inputs):
        hadamard_product = tf.math.multiply(self.mask, self.kernel)
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            output = tf.linalg.matmul(inputs, hadamard_product)
            shape = inputs.get_shape().as_list()
            output_shape = shape[:-1] + [self.units]
            output.set_shape(output_shape)
        else: 
            if not self._mixed_precision_policy.should_cast_variables:
                inputs = math_ops.cast(inputs, self.dtype)
            output = tf.linalg.matmul(inputs, hadamard_product)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=None)
        # pass the result of the layer through the activ. function
        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def set_mask(self, mask):
        """Sets the mask of the layer, from a Numpy array.
        Arguments:
            mask: a Numpy array. Its shape must match
                that of the mask of the layer.
        """
        self.mask.assign(value=mask)


class AddWithBias(Layer):
    # these are parameters that can be passed in to the creation of a new
    #   instance, with their default values
    def __init__(self, units,
                 activation='sigmoid',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        # each instance of the class will have all of these attributes
        self.activation = activations.get(activation)
        self.units = units
        self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=2)]
        super(AddWithBias, self).__init__(**kwargs)

    def build(self, input_shape, ):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        # note here how the weights matrix dimensions are equal to those of the
        #   mask
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=initializers.get('zeros'),
                                    name='bias')

        self.input_spec = [InputSpec(min_ndim=2, axes={-1: input_dim}), 
                           InputSpec(min_ndim=2, axes={-1: input_dim})]
        self.built = True
        super(AddWithBias, self).build(input_shape)

    # this function implements the layer's computations
    def call(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        output += self.bias
        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
