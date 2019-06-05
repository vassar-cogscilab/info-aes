"""
NOTES TO SELF 6/5 5:38 PM
–– figure out how the layer will pass its indices to the next layer
–– the output parameter should be associated somehow with the direct connection...
–– learn more about mini-batch mask randomization
"""

# implements a masked version of the keras Dense layer to be used as a hidden
#   or an output layer in MADE
class MaskedDense(Layer):
    # these are parameters that can be passed in to the creation of a new instance, with their default values
    def __init__(self, units,
                 data_features,
                 is_output=False,
                 input_indices=[],
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MaskedDense, self).__init__(**kwargs)

        # each instance of the class will have all of these attributes

        self.data_features = data_features
        self.units = units
        self.is_output = is_output
        self.input_indices = input_indices
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.input_spec = InputSpec(min_ndim=2)

        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        def gen_mask(data_features, units, is_output, input_indices, input_dim):
            mask = np.zeros((units, input_dim))
            # if input indices are not supplied, we must generate a list
            if len(input_indices) == 0:
                for i in range(input_dim):
                    input_indices.append(i+1)

            # we must generate a list of node input_indices
            layer_indices = []
            # if the layer is output, these indices should list the nodes in
            #   order
            for i in range(units):
                layer_indices.append(i+1)
            # otherwise, these indices should
            if is_output == False:
                for i in range(units):
                    layer_indices[i] = np.random.randint(1, data_features - 1)

            print(layer_indices)
            print(input_indices)
            # populate mask with appropriate values
            for i in range(len(layer_indices)): # iterate over every layer node
                for j in range(len(input_indices)): # iterate over every input node
                    if is_output:
                        if input_indices[j] < layer_indices[i]: # if the input index > the index
                            mask[i][j] = 1
                    else:
                        if input_indices[j] <= layer_indices[i]:
                            mask[i][j] = 1
            return (mask, layer_indices)

        mask, layer_indices = gen_mask(data_features, units, is_output, input_indices, input_dim)
        self.mask = mask
        self.layer_indices = layer_indices

    def build(self, input_shape, ):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            # bias is a vector, hence the shape argument here
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    # this function implements the layer's computations
    def call(self, inputs):
        output = K.dot(inputs, K.multiply(self.mask, self.kernel))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        # pass the result of the layer through the activ. function
        if self.activation is not None:
            output = self.activation(output)
        return output

    # not sure what's goin on here but i know it's important
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
