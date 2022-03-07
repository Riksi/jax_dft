import functools

import jax
from jax import lax
from jax import nn
from jax.experimental import stax
import jax.numpy as jnp
from jax.scipy import ndimage

from jax_dft import scf
from jax_dft import utils

_STAX_ACTIVATION = {
    'relu': stax.Relu,
    'elu': stax.Elu,
    'softplus': stax.Softplus,
    'swish': stax.elementwise(nn.swish)
}


def negativity_transform():
    """Layer construction for negativity transform

    Used as last layer for xc network
    since correlation and exchange must be negative
    according to exact conditions

    Returns:
        (init_fn, apply_fn) pair

    """
    def negative_fn(x):
        return -nn.swish(x)

    return stax.elementwise(negative_fn)


def _exponential_function(displacements, width):
    return jnp.exp(
        -jnp.abs(displacements) / width
    ) / (2 * width)


def _exponential_function_channels(displacements, widths):
    """

    Args:
        displacements: [spatial_size, spatial_size]
        widths: [num_channels]

    Returns:

    """
    return jax.vmap(_exponential_function,
                    in_axes=(None, 0),
                    out_axes=2)(displacements, widths)

def exponential_global_convolution(
    num_channels,
    grids,
    minval,
    maxval,
    downsample_factor=0,
    eta_init=nn.initializers.normal()
):
    grids = grids[:: 2 ** downsample_factor]
    # [[g1, g2, ..., gN]] - [[g1], [g2], ..., [gN]]
    # = [[0, g2 - g1, ..., gN - g1], ..., [g1 - gN, g2 - gN, ..., 0]]
    # -> displacements[x, y] = grids[y] - grids[x]
    displacements = jnp.expand_dims(grids, axis=0) - jnp.expand_dims(
        grids, axis=1
    )
    dx = utils.get_dx(grids)

    def init_fn(rng, input_shape):
        if num_channels <= 0:
            raise ValueError(f'num_channels must be positive but got {num_channels}')
        if len(input_shape) != 3:
            raise ValueError(f'The ndim of input shape should be 3, but got {len(input_shape)}')
        if input_shape[1] != len(grids):
            raise ValueError(f'input_shape[1] should be len(grids), but got {input_shape[1]}')
        if input_shape[2] != 1:
            raise ValueError(f'input_shape[2] should be 1, but got {input_shape[2]}')

        output_shape = input_shape[:-1] + (num_channels,)
        eta = eta_init(rng, (num_channels,))
        return output_shape, (eta,)

    def apply_fn(params, inputs, **kwargs):
        """Applies layer

        Args:
            params: (eta,)
            inputs: [batch_size, num_grids, num_in_channels]
            **kwargs: Unused

        Returns:
            Float numpy array with shape [batch_size, num_grids, num_channels]
        """
        del kwargs
        (eta,) = params
        # [num_grids, num_grids, num_in_channels]
        kernels = _exponential_function_channels(displacements,
                                                 minval + (maxval - minval) *
                                                 nn.sigmoid(eta))

        # Outputs a weighted sum of inputs where the weight
        # applied to position y of the input for position x of the output
        # depends on the displacement between y and x
        # I think it is like attention where the weight is higher
        # when y is close to x and exponentially decreases otherwise
        # with rate controlled by the width which is parameterised by eta

        # displacements[x, y] = grids[y] - grids[x]
        # kernels[x, y] results from a function applied
        # elementwise to abs(displacements[x, y])
        # => output[b, x] = sum_y(inputs[b, y] * kernels[x, y])
        # = sum_y(inputs[b, y] * fn(|grids[y] - grids[x])|)

        # inputs: [batch_size, num_grids, num_in_channels]
        # kernels: [num_grids, num_grids, num_channels]
        return jnp.squeeze(
            # In init_fn we ensure num_in_channels=1
            # [batch_size, num_in_channels, num_grids, num_channels]
            jnp.tensordot(
                inputs, kernels, axes=(1, 0)
            ) * dx,
            axis=1
        )

    return init_fn, apply_fn


def global_conv_block(num_channels, grids,
                      minval, maxval,
                      downsample_factor):
    """Global convolution block

    Downsample, apply global conv, upsample, concatenate with input

    Args:
        num_channels:
        grids:
        minval:
        maxval:
        downsample_factor:

    Returns:

    """

    layers = []
    layers.extend([linear_interpolation_transpose()] * downsample_factor)
    layers.extend([exponential_global_convolution(
        # one channel for input
        num_channels - 1,
        grids,
        minval,
        maxval,
        downsample_factor
    )
    ])
    layers.extend([linear_interpolation()] * downsample_factor)

    global_conv_path = stax.serial(layers)

    return stax.serial(
        stax.FanOut(2),
        stax.parallel(stax.Identity, global_conv_path),
        stax.FanInConcat(axis=-1)
    )


def self_interaction_weight(reshaped_density, dx, width):
    """

    Args:
        reshaped_density: any shape with total size of num_grids
        dx:
        width:

    Returns:

    """
    # Does this approximates i.e. \int \sum_i |\phi_i|^2 dz
    # (where i includes spin as well as position)?
    # But what is the meaning of that since
    # this would be approximately the number of electrons?
    # The tests include the possibility that reshaped_density
    # sums to a fractional value as well as integer values.
    # The tests seem to assume that the shape of the
    # input will be [-1, N, 1].
    density_integral = jnp.sum(reshaped_density) * dx
    return jnp.exp(-jnp.square((density_integral - 1) / width))


def self_interaction_layer(grids, interaction_fn):
    """

    Args:
        grids:
        interaction_fn:

    Returns:

    """
    dx = utils.get_dx(grids)

    def init_fn(rng, input_shape):
        if len(input_shape) != 2:
            raise ValueError(f"self_interaction_layer must have two inputs"
                             f" but got {len(input_shape)}")
        if input_shape[0] != input_shape[1]:
            raise ValueError(f"The input self_interaction_layer must consist of two identical shapes"
                             f" but got {input_shape[0]} and {input_shape[1]}")

        return input_shape[0], (jnp.array(1.),)

    def apply_fn(params, inputs, **kwargs):
        (width,) = params
        reshaped_density, features = inputs
        beta = self_interaction_weight(
            reshaped_density=reshaped_density,
            dx=dx,
            width=width
        )
        hartree = -0.5 * scf.get_hartree_potential(
            # [B, G, ...] -> [B * G * ...]
            reshaped_density.reshape(-1), grids, interaction_fn
        ).reshape(reshaped_density.shape)  # [B * G * ...] -> [B, G, ...]
        return hartree * beta + features * (1 - beta)

    return init_fn, apply_fn


def wrap_network_with_self_interaction_layer(network, grids, interaction_function):
    return stax.serial(
        stax.FanOut(2),
        stax.parallel(stax.Identity, network),
        self_interaction_layer(grids, interaction_function)
    )


def GeneralConvWithoutBias(
    dimension_numbers, out_chan, filter_shape,
    strides=None, padding='VALID', W_init=None
):
    lhs_spec, rhs_spec, _ = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    # For rhs_spec=HIO arguments are 2, 3
    W_init = W_init or jax.nn.initializers.he_normal(
        rhs_spec.index('I'), rhs_spec.index('O')
    )

    def init_fun(rng, input_shape):
        filter_shape_iter = iter(filter_shape)
        # For lhs_spec=NHC, rhs_spec=HIO you get
        # [filter_shape[0], input_shape[2], out_chan]
        kernel_shape = [
            out_chan if c == 'O' else input_shape[lhs_spec.index('C')] if c == 'I' else next(filter_shape_iter)
            for c in rhs_spec
        ]
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers
        )
        W = W_init(rng, kernel_shape)
        return output_shape, (W,)

    def apply_fun(params, inputs, **kwargs):
        del kwargs
        W, = params
        return lax.conv_general_dilated(
            inputs, W, strides, padding, one, one, dimension_numbers=dimension_numbers
        )

    return init_fun, apply_fun


Conv1D = functools.partial(
    GeneralConvWithoutBias, ('NHC', 'HIO', 'NHC')
)


def _resample_1d(inputs, new_size):
    if inputs.ndim != 1:
        raise ValueError(f'inputs must be 1d but has shape {inputs.shape}')
    x = jnp.linspace(0, inputs.size - 1, num=new_size)
    # N = new_size
    # s = (inputs.size - 1) / (N - 1)
    # x = [0, s, 2*s, ..., (N - 1)*s] = [0, s, 2*s, ..., inputs.size - 1]
    # The following samples by linearly interpolating x[i]
    # between input[floor(x[i])] and input[ceil(x[i])]
    # where 0 <= floor(x[i]) <= ceil(x[i]) <= inputs.size - 1
    # so they are valid indices into inputs

    # outputs[i] = inputs[floor(x[i])] + (inputs[ceil(x[i])] - inputs[floor(x[i])]) * (x[i] - floor(x[i]))

    return ndimage.map_coordinates(
        inputs, [x], order=1, mode='nearest'
    )


#  linear_interpolation
def linear_interpolation():
    """
    Input has shape NHC where the H is the spatial dimension
    and it is upsampled to 2*m - 1

    Returns:

    """
    def init_fn(rng, input_shape):
        del rng
        output_shape = input_shape[0], 2 * input_shape[1] - 1, input_shape[2]
        return output_shape, ()

    def apply_fn(params, inputs, **kwargs):
        del params, kwargs
        upsample = functools.partial(
            _resample_1d, new_size=2 * inputs.shape[1] - 1
        )
        return jax.vmap(jax.vmap(upsample, 0, 0), 2, 2)(inputs)

    return init_fn, apply_fn


def _with_edge(x, scale):
    return jnp.concatenate(
        [x[:1] * scale, x[1:-1] * scale, x[-1:] * scale]
    )




def linear_interpolation_transpose():
    """
    Input has shape NHC where the H is the spatial dimension
    and it is downsampled from 2*m - 1 to m
    Returns:

    """

    def init_fn(rng, input_shape):
        del rng
        if (input_shape[1] % 2) == 0:
            raise ValueError(f'input_shape[1] must be an odd number, got {input_shape[1]}')
        m = (input_shape[1] + 1) // 2
        output_shape = (input_shape[0], m, input_shape[2])
        return output_shape, ()

    def apply_fn(params, inputs, **kwargs):
        del params, kwargs
        m = (inputs.shape[1] + 1)//2
        upsample = functools.partial(
            _resample_1d, new_size=inputs.shape[1]
        )
        dummy = jnp.zeros((m,), dtype=inputs.dtype)
        _, vjp_fn = jax.vjp(upsample, dummy)

        def downsample(x):
            # Since the output elements of `linear_interpolate`
            # are just linear combination
            # of input elements, the loss gradient with respect to
            # input[i] is just the weighted sum of the output gradients
            # to which input[i] contributes which are
            #   output[2*i] with weight of 1
            #   output[2*i - 1] with weight of 0.5, i > 0
            #   output[2*i + 1] with weight of 0.5, i < (len(input) - 1)

            # I think that x contains the downstream gradients
            # with respect to the outputs i.e. x[i] = dL/d(out[i])

            # Letting f = vjp_fn(x) the gradients are as follows
            #   Non-edge i.e. 0 < i < (len(f) - 1)
            #       f[i] = x[i] + (x[i - 1] + x[i + 1]) / 2
            #   Edge
            #       i = 0: f[i] = x[i] + x[i + 1] / 2
            #       i = (len(f) - 1): f[i] = x[i] + x[i - 1] / 2

            # A non-edge element i gets a full contribution from x[i] and
            # half a contribution from each of x[i - 1] and x[i + 1]
            # so from 2 elements in total so maybe you say that since this
            # scales as 2 x a single activation you need to scale it by 0.5

            # Originally the edge elements only get contributions
            # from 1.5 elements, or after the scaling step 0.75 elements
            # so you add a 0.25 of the edge gradient to make that up to 1.

            output = 0.5 * vjp_fn(x)[0]
            output = output.at[:1].add(0.25 * x[:1])
            output = output.at[-1:].add(0.25 * x[-1:])
            return output

        return jax.vmap(jax.vmap(
                downsample, 0, 0
            ), 2, 2)(inputs)

    return init_fn, apply_fn


def upsampling_block(num_filters, activation):
    """
    Input with spatial dimension `m` upsampled to `2 * m - 1` followed by convolution with `num_filters`.

    Args:
        num_filters:
        activation:

    Returns:

    """
    return stax.serial(
        linear_interpolation(),
        Conv1D(num_filters, filter_shape=(3,), padding='SAME'),
        _STAX_ACTIVATION[activation]
    )


def downsampling_block(num_filters, activation):
    """
    Input with spatial dimension `m` first sent through
    a convolution layer with `num_filters` then downsampled to size `(m + 1) / 2`

    Args:
        num_filters:
        activation:

    Returns:

    """
    return stax.serial(
        Conv1D(num_filters, filter_shape=(3,), padding='SAME'),
        linear_interpolation_transpose(),
        _STAX_ACTIVATION[activation]
    )


def _build_unet_shell(layer, num_filters, activation):
    return stax.serial(
        downsampling_block(num_filters, activation=activation),
        stax.FanOut(2),
        stax.parallel(stax.Identity, layer),
        stax.FanInConcat(axis=-1),
        upsampling_block(num_filters, activation=activation)
    )


def build_unet(
    num_filters_list, core_num_filters, activation,
    num_channels=0, grids=None, minval=None, maxval=None,
    apply_negativity_transform=True
):
    layer = stax.serial(
        Conv1D(core_num_filters, filter_shape=(3,), padding='SAME'),
        _STAX_ACTIVATION[activation],
        Conv1D(core_num_filters, filter_shape=(3,), padding='SAME'),
        _STAX_ACTIVATION[activation]
    )

    for num_filters in num_filters_list[::-1]:
        layer = _build_unet_shell(layer, num_filters, activation)

    layer = stax.serial(
        layer,
        Conv1D(1, filter_shape=(1,), padding='SAME')
    )

    layers_before_network = []
    if num_channels > 0:
        layers_before_network.append(
            exponential_global_convolution(
                num_channels, grids=grids, minval=minval, maxval=maxval
            )
        )

    if apply_negativity_transform:
        return stax.serial(
            *layers_before_network,
            layer,
            negativity_transform()
        )

    else:
        return stax.serial(
            *layers_before_network,
            layer
        )


def build_global_local_conv_net(
    num_global_filters,
    num_local_filters,
    num_local_conv_layers,
    activation,
    grids,
    minval,
    maxval,
    downsample_factor,
    apply_negativity_transform
):
    layers = []
    layers.append(
        global_conv_block(
            num_channels=num_global_filters,
            grids=grids,
            minval=minval,
            maxval=maxval,
            downsample_factor=downsample_factor))
    layers.extend([
        Conv1D(num_local_filters, filter_shape=(3,), padding='SAME'),
        _STAX_ACTIVATION[activation]] * num_local_conv_layers
    )

    layers.append(
        Conv1D(1, filter_shape=(1,), padding='SAME')
    )

    if apply_negativity_transform:
        layers.append(negativity_transform())

    return stax.serial(*layers)


def build_sliding_net(
    window_size,
    num_filters_list,
    activation,
    apply_negativity_transform=True
):

    if window_size < 1:
        raise ValueError(f'window size cannot be less than 1 but got {window_size}')

    layers = []
    for i, num_filters in enumerate(num_filters_list):
        if i == 0:
            filter_shape = (window_size,)
        else:
            filter_shape = (1,)
        layers.extend(
           [Conv1D(
                num_filters,
                filter_shape=filter_shape,
                padding='SAME'
            ),
            _STAX_ACTIVATION[activation]]
        )

    layers.append(
        Conv1D(
            1, filter_shape=(1,), padding='SAME'
        )
    )

    if apply_negativity_transform:
        layers.append(negativity_transform())

    return stax.serial(*layers)


def _check_network_output(output, num_features):
    shape = output.shape
    if output.ndim != 2 or shape[1] != num_features:
        raise ValueError(
            f'The output should have shape (-1, {num_features})'
            f' but got {shape}'
        )


def _is_power_of_two(number):
    """
    If `number` is non zero is 2^a
    then binary representation of `number`
    is 1 followed by a-1 zeros whilst `number-1`
    is a-1 ones so if "AND" these the result is 0.

        10...00
    AND 01...11
        -------
        00...00

    Args:
        number:

    Returns:

    """
    return number and not number & (number - 1)


def _spatial_shift_input(features, num_spatial_shift):
    """

    Args:
        features: Float numpy array with shape
            (batch_size, num_grids, num_features)
        num_spatial_shift:

    Returns:
        Float numpy array with shape (batch_size * num_spatial_shift, num_grids, num_features)
    """

    output = []
    for sample_feature in features:
        for offset in range(num_spatial_shift):
            output.append(
                jax.vmap(
                functools.partial(utils.shift, offset=offset)
                    , 1, 1)(sample_feature)
            )
    return jnp.stack(output)


def _reverse_spatial_shift_output(array):
    """

    Args:
        array: Float numpy array with shape
            (num_spatial_shift, num_features)

    Returns:

    """
    output = []
    for offset, sample_array in enumerate(array):
        output.append(
            utils.shift(sample_array, offset=-offset)
        )
    return jnp.stack(output)


def local_density_approximation(network):
    network_init_fn, network_apply_fn = network

    def init_fn(rng):
        _, params = network_init_fn(rng, (-1, 1))
        return params

    @jax.jit
    def xc_energy_density_fn(density, params):
        """LDA parameterised by neural network

        Takes density as input and output is of size (-1, 1).

        Args:
            params:
            inputs: (num_grids,)
            **kwargs:

        Returns:
            Float numpy array of shape (num_grids,)

        """
        # Network is applied to each point of the density on the grid
        # so can think of the input as having batch size of `num_grids`
        # and a single feature per batch element
        output = network_apply_fn(params, jnp.expand_dims(density, 1))
        _check_network_output(output, num_features=1)
        output = jnp.squeeze(output, axis=-1)
        return output

    return init_fn, xc_energy_density_fn


def global_functional(network, grids, num_spatial_shift=1):
    """Functional with global density information parameterised by neural network

    Input which comprises the entire density
    is expanded along the batch dimension by shifting if num_spatial_shift > 1,
    followed by a convolutional network.

    Two types of mapping are possible

    * many-to-one *
        - Using a network with an input window size of 2 * r + 1
            followed by filter size of 1 (as in sliding_net about) you use the density
            of the r neighbours on each side to predict the XC energy density
        - When r=0, this is like LDA where only the value of density at a point is used
        - Using r=1 is like using finite difference as in GGA
        - With large r you can think of it as non-local functional

    * many-to-many *
        - Here density at all points is used to predict the density at any point
        - A U-Net type architecture is used to capture both low and high-level
            features from the input

    Args:
        network:
        grids: float numpy array of shape (num_grids,)
        num_spatial_shift:

    Returns:

    """

    if num_spatial_shift < 1:
        raise ValueError(f"num_spatial_shift must be at least 1, but got {num_spatial_shift}")

    network_init_fn, network_apply_fn = network
    num_grids = grids.shape[0]

    if not _is_power_of_two(num_grids - 1):
        raise ValueError(f"num_grids must be a power of 2 plus 1"
                         f" for global functional, but got {num_grids}")

    def init_fn(rng):
        _, params = network_init_fn(rng, (-1, num_grids, 1))
        return params

    @jax.jit
    def xc_energy_density_fn(density, params):
        """

        Args:
            density: float numpy array of shape (num_grids,)
            params:

        Returns:
            Float numpy array of shape (num_grids,)

        """
        # (1, num_grids, 1)
        input_features = density[jnp.newaxis, :, jnp.newaxis]
        # (1, num_grids, 1) -> (num_spatial_shift, num_grids, 1)
        if num_spatial_shift > 1:
            input_features = _spatial_shift_input(density, num_spatial_shift=num_spatial_shift)

        output = network_apply_fn(params, input_features)

        # (num_spatial_shift, num_grids, 1) -> ((num_spatial_shift, num_grids)
        output = jnp.squeeze(output, axis=2)
        _check_network_output(output, num_grids)
        if num_spatial_shift > 1:
            output = _reverse_spatial_shift_output(output)

        output = jnp.mean(output, axis=0)
        return output

    return init_fn, xc_energy_density_fn







#  global_functional
#  xc_energy_density_fn


