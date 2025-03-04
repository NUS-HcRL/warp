# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example jax_callable()
#
# Examples of calling annotated Python functions from JAX.
###########################################################################

from functools import partial

import jax
import jax.numpy as jnp

import warp as wp
from warp.jax_experimental.ffi import jax_callable


@wp.kernel
def scale_kernel(a: wp.array(dtype=float), s: float, output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = a[tid] * s


@wp.kernel
def scale_vec_kernel(a: wp.array(dtype=wp.vec2), s: float, output: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    output[tid] = a[tid] * s


# The Python function to call.
# Note the argument annotations, just like Warp kernels.
def example_func(
    # inputs
    a: wp.array(dtype=float),
    b: wp.array(dtype=wp.vec2),
    s: float,
    # outputs
    c: wp.array(dtype=float),
    d: wp.array(dtype=wp.vec2),
):
    wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[c])
    wp.launch(scale_vec_kernel, dim=b.shape, inputs=[b, s], outputs=[d])


def example1():
    jax_func = jax_callable(example_func, num_outputs=2, vmap_method="broadcast_all")

    @jax.jit
    def f():
        # inputs
        a = jnp.arange(10, dtype=jnp.float32)
        b = jnp.ones((5, 2), dtype=jnp.float32)  # wp.vec2
        s = 2.0

        # output shapes
        output_dims = {"c": a.shape, "d": b.shape}

        c, d = jax_func(a, b, s, output_dims=output_dims)

        return c, d

    r1, r2 = f()
    print(r1)
    print(r2)


def example2():
    jax_func = jax_callable(example_func, num_outputs=2, vmap_method="broadcast_all")

    # NOTE: scalar arguments must be static compile-time constants
    @partial(jax.jit, static_argnames=["s"])
    def f(a, b, s):
        # output shapes
        output_dims = {"c": a.shape, "d": b.shape}

        c, d = jax_func(a, b, s, output_dims=output_dims)

        return c, d

    # inputs
    a = jnp.arange(10, dtype=jnp.float32)
    b = jnp.ones((5, 2), dtype=jnp.float32)  # wp.vec2
    s = 2.0

    r1, r2 = f(a, b, s)
    print(r1)
    print(r2)


def main():
    wp.init()
    wp.load_module(device=wp.get_device())

    examples = [example1, example2]

    for example in examples:
        print("\n===========================================================================")
        print(f"{example.__name__}:")
        example()


if __name__ == "__main__":
    main()
