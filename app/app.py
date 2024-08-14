import warp as wp
import jax
import jax.numpy as jp

# import experimental feature
from warp.jax_experimental import jax_kernel

# kernel with multiple inputs and outputs
@wp.kernel
def multiarg_kernel(
    # inputs
    a: wp.array(dtype=float),
    b: wp.array(dtype=float),
    c: wp.array(dtype=float),
    # outputs
    ab: wp.array(dtype=float),
    bc: wp.array(dtype=float),
):
    tid = wp.tid()
    ab[tid] = a[tid] + b[tid]
    bc[tid] = b[tid] + c[tid]

# create a Jax primitive from a Warp kernel
jax_multiarg = jax_kernel(multiarg_kernel)

# use the Warp kernel in a Jax jitted function with three inputs and two outputs
@jax.jit
def f():
    a = jp.full(64, 1, dtype=jp.float32)
    b = jp.full(64, 2, dtype=jp.float32)
    c = jp.full(64, 3, dtype=jp.float32)
    return jax_multiarg(a, b, c)

x, y = f()

print(x)
print(y)