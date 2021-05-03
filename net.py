from jax import jit, lax, nn
from jax import numpy as jnp
from jax import random as jrand
from jax.experimental import stax

from args import args


# Modified from stax.GeneralConv
def MaskedConv2d(out_chan, filter_shape, dilation, exclusive):
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    kernel_h, kernel_w = filter_shape
    dilation_h, dilation_w = dilation
    one = (1, 1)
    W_init = nn.initializers.glorot_normal(rhs_spec.index('I'),
                                           rhs_spec.index('O'))
    b_init = nn.initializers.normal(args.eps)

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
            window_strides=one,
            padding='VALID',
            lhs_dilation=one,
            rhs_dilation=dilation,
            dimension_numbers=dimension_numbers,
        )
        out += b
        return out

    return init_fun, apply_fun


# The network should be shallow, so there is no need for residue blocks
# Inputs in {-1, 1}
def get_net():
    layers = []
    dilation = 1

    for i in range(args.net_depth):
        if i > 0:
            layers.append(stax.elementwise(nn.silu))
        layers.append(
            MaskedConv2d(
                1 if i == args.net_depth - 1 else args.net_width,
                ((args.kernel_size + 1) // 2, args.kernel_size),
                (dilation, dilation),
                exclusive=(i == 0),
            ))
        dilation += args.dilation_step

    # Outputs in (0, 1)
    layers.append(stax.Sigmoid)

    net_init, net_apply = stax.serial(*layers)
    net_apply = jit(net_apply)
    return net_init, net_apply


# Test if the network is autoregressive
def test_autoreg():
    batch_size = 4
    # Index of the changed sample in the batch
    B = 1

    net_init, net_apply = get_net()

    rng_net, rng_spins = jrand.split(jrand.PRNGKey(args.seed))
    in_shape = (batch_size, args.L, args.L, 1)
    out_shape, init_params = net_init(rng_net, in_shape)

    spins = jrand.bernoulli(rng_spins, shape=in_shape).astype(
        jnp.float32) * 2 - 1
    s_hat = net_apply(init_params, spins)

    for i in range(args.L):
        for j in range(args.L):
            # Try to change input elements, one at a time
            spins_new = spins.at[B, i, j, 0].set(-spins[B, i, j, 0])
            s_hat_new = net_apply(init_params, spins_new)
            diff = (s_hat_new - s_hat)[:, :, :, 0]

            # The later output elements will not change
            for jj in range(j + 1, args.L):
                diff = diff.at[B, i, jj].set(0)
            for ii in range(i + 1, args.L):
                for jj in range(args.L):
                    diff = diff.at[B, ii, jj].set(0)
            print(i, j, jnp.allclose(diff, 0))


if __name__ == '__main__':
    test_autoreg()
