# The score function (REINFORCE) gradient estimator of an expectation

from functools import partial

from jax import custom_vjp, grad
from jax import numpy as jnp
from jax import vmap


# log_prob_args and expected_args are independent from params when taking the
# gradient. They can be continuous or discrete, and they can be pytrees
# Does not support higher-order derivatives
@partial(custom_vjp, nondiff_argnums=(0, 1, 5))
def expect(log_prob_fun,
           expected_fun,
           params,
           log_prob_args,
           expected_args,
           mean_grad_expected_is_zero=False):
    ys = expected_fun(params, expected_args)
    y_mean = ys.mean(axis=0)
    return y_mean


def expect_fwd(log_prob_fun, expected_fun, params, log_prob_args,
               expected_args, mean_grad_expected_is_zero):
    ys = expected_fun(params, expected_args)
    y_mean = ys.mean(axis=0)

    # Use the baseline trick to reduce the variance
    weight = ys - y_mean

    return y_mean, (params, log_prob_args, expected_args, weight)


def expect_bwd(log_prob_fun, expected_fun, mean_grad_expected_is_zero, res, g):
    params, log_prob_args, expected_args, weight = res

    def f(params):
        log_p = log_prob_fun(params, log_prob_args)
        out = vmap(jnp.multiply)(weight, log_p)
        if not mean_grad_expected_is_zero:
            out += expected_fun(params, expected_args)
        out = out.mean(axis=0)
        out = (g * out).sum()
        return out

    grad_params = grad(f)(params)

    # Gradients of log_prob_args and expected_args are None
    return grad_params, None, None


expect.defvjp(expect_fwd, expect_bwd)
