# The score function (REINFORCE) gradient estimator of an expectation

from functools import partial

from jax import custom_vjp, grad
from jax import numpy as jnp
from jax import vmap


# log_prob_args and expected_args are independent of params when taking the
# gradient. They can be continuous or discrete, and they can be pytrees
# Does not support higher-order derivatives yet
@partial(custom_vjp, nondiff_argnums=(0, 1))
def expect(log_prob_fun, expected_fun, params, log_prob_args, expected_args):
    ys = expected_fun(params, expected_args)
    y_mean = ys.mean(axis=0)
    return y_mean


def expect_fwd(log_prob_fun, expected_fun, params, log_prob_args,
               expected_args):
    ys = expected_fun(params, expected_args)
    y_mean = ys.mean(axis=0)

    # Use the baseline trick to reduce the variance
    weight = ys - y_mean

    return y_mean, (params, log_prob_args, expected_args, weight)


# TODO: in principle, the gradient of an expectation is another expectation,
# so it should support higher-order derivatives
# But I don't know how to transform log_prob_fun into grad(log_prob_fun) while
# keeping the batch dimension and without a loop through the batch dimension
def expect_bwd(log_prob_fun, expected_fun, res, g):
    params, log_prob_args, expected_args, weight = res

    def f(params):
        log_p = log_prob_fun(params, log_prob_args)
        term1 = vmap(jnp.multiply)(weight, log_p)
        term2 = expected_fun(params, expected_args)
        out = (term1 + term2).mean(axis=0)
        out = (g * out).sum()
        return out

    grad_params = grad(f)(params)

    # Gradients of log_prob_args and expected_args are None
    return grad_params, None, None


expect.defvjp(expect_fwd, expect_bwd)
