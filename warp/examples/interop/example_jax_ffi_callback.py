# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example register_ffi_callback()
#
# Examples of calling arbitrary Python functions from JAX.
###########################################################################

import jax
import jax.numpy as jnp
import numpy as np

import warp as wp
from warp.jax import get_jax_device
from warp.jax_experimental.ffi import register_ffi_callback


@wp.kernel
def scale_kernel(a: wp.array(dtype=float), s: float, output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = a[tid] * s


@wp.kernel
def scale_vec_kernel(a: wp.array(dtype=wp.vec2), s: float, output: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    output[tid] = a[tid] * s


def example1():
    # the Python function to call
    def print_args(inputs, outputs, attrs, ctx):
        def buffer_to_string(b):
            return str(b.dtype) + str(list(b.shape)) + " @%x" % b.data

        print("Inputs:     ", ", ".join([buffer_to_string(b) for b in inputs]))
        print("Outputs:    ", ", ".join([buffer_to_string(b) for b in outputs]))
        print("Attributes: ", "".join(["\n  %s: %s" % (k, str(v)) for k, v in attrs.items()]))

    # register callback
    register_ffi_callback("print_args", print_args)

    # set up call
    call = jax.ffi.ffi_call("print_args", jax.ShapeDtypeStruct((1, 2, 3), jnp.int8))

    # call it
    call(
        jnp.arange(16),
        jnp.arange(32.0).reshape((4, 8)),
        str_attr="hi",
        f32_attr=np.float32(4.2),
        dict_attr={"a": 1, "b": 6.4},
    )


def example2():
    # the Python function to call
    def warp_func(inputs, outputs, attrs, ctx):
        # input arrays
        a = inputs[0]
        b = inputs[1]

        # scalar attributes
        s = attrs["scale"]

        # output arrays
        c = outputs[0]
        d = outputs[1]

        device = wp.device_from_jax(get_jax_device())
        stream = wp.Stream(device, cuda_stream=ctx.stream)

        with wp.ScopedStream(stream):
            # launch with arrays of scalars
            wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[c])

            # launch with arrays of vec2
            # NOTE: the input shapes are from JAX arrays, we need to strip the inner dimension for vec2 arrays
            wp.launch(scale_vec_kernel, dim=b.shape[0], inputs=[b, s], outputs=[d])

    # register callback
    register_ffi_callback("warp_func", warp_func)

    n = 10

    # inputs
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones((n // 2, 2), dtype=jnp.float32)  # wp.vec2
    s = 2.0

    # set up call
    out_types = [
        jax.ShapeDtypeStruct(a.shape, jnp.float32),
        jax.ShapeDtypeStruct(b.shape, jnp.float32),  # wp.vec2
    ]
    call = jax.ffi.ffi_call("warp_func", out_types)

    # call it
    c, d = call(a, b, scale=s)

    print(c)
    print(d)


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
