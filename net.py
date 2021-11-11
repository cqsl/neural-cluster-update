from jax import lax, nn
from jax import numpy as jnp
from jax import random as jrand
from jax.example_libraries import stax

from args import args


def prev_index_2d(i, j, L):
    v = i * L + j - 1
    i = v // L
    j = v % L
    return i, j


# Modified from stax.GeneralConv
# init_cache_fun and apply_fast_fun are for the fast autoregressive sampling,
# which is a drop-in replacement for the regular autoregressive sampling
def MaskedConv2d(out_chan, filter_shape, dilation, exclusive):
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    lhs_spec, rhs_spec, out_spec = dimension_numbers

    kernel_h, kernel_w = filter_shape
    dilation_h, dilation_w = dilation
    recep_h = (kernel_h - 1) * dilation_h + 1
    recep_w = (kernel_w - 1) * dilation_w + 1

    W_init = nn.initializers.glorot_normal(rhs_spec.index('I'),
                                           rhs_spec.index('O'))
    b_init = nn.initializers.zeros

    mask = jnp.ones((kernel_h, kernel_w, 1, 1))
    mask = mask.at[-1, kernel_w // 2 + (not exclusive):].set(0)

    def init_fun(rng, in_shape):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [
            out_chan if c == 'O' else in_shape[lhs_spec.index('C')]
            if c == 'I' else next(filter_shape_iter) for c in rhs_spec
        ]
        out_shape = [
            out_chan if c == 'C' else in_shape[lhs_spec.index(c)]
            for c in out_spec
        ]
        bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]

        rng_W, rng_b = jrand.split(rng)
        W = W_init(rng_W, kernel_shape)
        W *= mask
        b = b_init(rng_b, bias_shape)

        return out_shape, (W, b)

    def init_cache_fun(params, inputs, index, **kwargs):
        out = apply_fun(params, inputs, **kwargs)

        index_h, _ = index

        cache_shape = [
            recep_h if c == 'H' else inputs.shape[lhs_spec.index(c)]
            for c in lhs_spec
        ]

        # Zero padding
        inputs = jnp.pad(inputs, (
            (0, 0),
            (recep_h, 0),
            (0, 0),
            (0, 0),
        ))

        cache = lax.dynamic_slice(inputs, (0, index_h + 1, 0, 0), cache_shape)
        return out, cache

    def apply_fun(params, inputs, **kwargs):
        W, b = params

        # Zero padding
        out = jnp.pad(inputs, (
            (0, 0),
            ((kernel_h - 1) * dilation_h, 0),
            (kernel_w // 2 * dilation_w, (kernel_w - 1) // 2 * dilation_w),
            (0, 0),
        ))

        out = lax.conv_general_dilated(
            out,
            mask * W,
            window_strides=(1, 1),
            padding='VALID',
            lhs_dilation=(1, 1),
            rhs_dilation=dilation,
            dimension_numbers=dimension_numbers,
        )
        out += b
        return out

    def apply_fast_fun(params, inputs, cache, index, **kwargs):
        W, b = params

        batch = inputs.shape[lhs_spec.index('N')]
        L = cache.shape[lhs_spec.index('W')]
        in_chan = inputs.shape[lhs_spec.index('C')]

        index_h, index_w = index
        if exclusive:
            index_h_in, index_w_in = prev_index_2d(index_h, index_w, L)
        else:
            index_h_in, index_w_in = index_h, index_w

        # First, update the cache

        def _add(cache):
            # return cache.at[:, -1, index_w_in, :].set(inputs)
            return lax.dynamic_update_slice(cache, inputs,
                                            (0, -1, index_w_in, 0))

        def _shift(cache):
            return jnp.concatenate(
                [
                    cache[:, 1:, :, :],
                    jnp.zeros((batch, 1, L, in_chan), dtype=cache.dtype)
                ],
                axis=1,
            )

        def _new_row(cache):
            return lax.cond(
                index_w_in == 0,
                lambda x: _add(_shift(x)),
                lambda x: _shift(_add(x)),
                cache,
            )

        def _update(cache):
            return lax.cond(index_w == 0, _new_row, _add, cache)

        cache = lax.cond(index_h_in >= 0, _update, lambda x: x, cache)

        # Then, use the cache to compute the outputs (the inputs are not used)

        # Zero padding
        cache_slice = jnp.pad(cache, (
            (0, 0),
            (0, 0),
            (kernel_w // 2 * dilation_w, (kernel_w - 1) // 2 * dilation_w),
            (0, 0),
        ))

        # cache = cache[:, :, index_w : index_w + recep_w, :]
        cache_slice = lax.dynamic_slice(cache_slice, (0, 0, index_w, 0),
                                        (batch, recep_h, recep_w, in_chan))

        out = lax.conv_general_dilated(
            cache_slice,
            mask * W,
            window_strides=(1, 1),
            padding='VALID',
            lhs_dilation=(1, 1),
            rhs_dilation=dilation,
            dimension_numbers=dimension_numbers,
        )
        assert out.shape == (batch, 1, 1, out_chan)
        out += b
        return out, cache

    return init_fun, apply_fun, init_cache_fun, apply_fast_fun


# Modified from stax.serial
def serial(*layers):
    def add_default_fast_funs(layer):
        if len(layer) == 4:
            return layer

        init_fun, apply_fun = layer
        return (
            init_fun,
            apply_fun,
            lambda params, inputs, index, **kwargs: (apply_fun(params, inputs),
                                                     ()),
            lambda params, inputs, cache, index, **kwargs:
            (apply_fun(params, inputs), ()),
        )

    layers = [add_default_fast_funs(layer) for layer in layers]
    init_funs, apply_funs, init_cache_funs, apply_fast_funs = zip(*layers)

    # kwargs is modified inplace
    def pop_rngs(kwargs):
        rng = kwargs.pop('rng', None)
        if rng is None:
            return (None, ) * len(layers)
        else:
            return jrand.split(rng, len(layers))

    def init_fun(rng, in_shape):
        params = []
        for fun in init_funs:
            rng, layer_rng = jrand.split(rng)
            in_shape, param = fun(layer_rng, in_shape)
            params.append(param)
        return in_shape, params

    def init_cache_fun(params, inputs, index, **kwargs):
        rngs = pop_rngs(kwargs)
        caches = []
        for fun, param, rng in zip(init_cache_funs, params, rngs):
            inputs, cache = fun(param, inputs, index, rng=rng, **kwargs)
            caches.append(cache)
        return inputs, caches

    def apply_fun(params, inputs, **kwargs):
        rngs = pop_rngs(kwargs)
        for fun, param, rng in zip(apply_funs, params, rngs):
            inputs = fun(param, inputs, rng=rng, **kwargs)
        return inputs

    def apply_fast_fun(params, inputs, caches, index, **kwargs):
        rngs = pop_rngs(kwargs)
        out_caches = []
        for fun, param, cache, rng in zip(apply_fast_funs, params, caches,
                                          rngs):
            inputs, cache = fun(param, inputs, cache, index, rng=rng, **kwargs)
            out_caches.append(cache)
        return inputs, out_caches

    return init_fun, apply_fun, init_cache_fun, apply_fast_fun


# The network should be shallow, so there is no need for residue blocks
# inputs: (batch, L, L, 1), values in {-1, 1}
# outputs: (batch, L, L, 1), values in (0, 1)
def get_net():
    layers = []
    dilation = 1

    for i in range(args.net_depth):
        if i > 0:
            layers.append(stax.Selu)
        layers.append(
            MaskedConv2d(
                1 if i == args.net_depth - 1 else args.net_width,
                ((args.kernel_size + 1) // 2, args.kernel_size),
                (dilation, dilation),
                exclusive=(i == 0),
            ))
        dilation += args.dilation_step

    layers.append(stax.Sigmoid)

    net_init, net_apply, net_init_cache, net_apply_fast = serial(*layers)
    return net_init, net_apply, net_init_cache, net_apply_fast
