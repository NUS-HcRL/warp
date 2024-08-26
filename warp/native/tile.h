#pragma once

#include "builtin.h"

#if !defined(__CUDA_ARCH__)
#define WP_TILE_SHARED static
#define WP_TILE_SYNC void
#else
#define WP_TILE_SHARED __shared__
#define WP_TILE_SYNC __syncthreads
#endif

// CUTLASS_PRAGMA_(UNROLL|NO_UNROLL) optimization directives for the CUDA compiler.
#if defined(__CUDA_ARCH__) && !defined(__INTELLISENSE__)
  #if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
    #define WP_PRAGMA_UNROLL _Pragma("unroll")
    #define WP_PRAGMA_NO_UNROLL _Pragma("unroll 1")
  #else
    #define WP_PRAGMA_UNROLL #pragma unroll
    #define WP_PRAGMA_NO_UNROLL #pragma unroll 1
  #endif

#else

    #define WP_PRAGMA_UNROLL
    #define WP_PRAGMA_NO_UNROLL

#endif

/* Tile Expressions

[x] Forward / Backward code-gen
[ ] wp.tile_map()
    [ ] Support user functions
    [ ] Support built-in functions
    [ ] Support for lambda functions
[ ] wp.tile_matmul()
    [ ] Forward
    [ ] Reverse
[ ] Support for n-d shape tiles / broadcasting / slicing?
[ ] Compile-time block dimensions
[ ] Support for CUB reductions
[ ] Support for CUB sorts
[ ] Examples
    [ ] GEMM
    [ ] Batched MLP
    [ ] Point cloud alignment
    [ ] Layer norm

*/

// wp.tile_load(A, offset, shape)
// wp.tile_load(A, (x, y), (16, 16))
// wp.tile_load(A, (x, y, z), (3, 3, 3))

// wp.tile_load(A, index, shape)
// wp.tile_load(A, x, m)
// wp.tile_load(A, x, y, m, n)
// wp.tile_load(A, x, y, z, m, n, o)
// wp.tile_load(A, x, y, z, m, n, o, p)

namespace wp
{

// Primary template
template <typename T, typename U>
struct is_same {
    static constexpr bool value = false;
};

// Specialization for the case when T and U are the same type
template <typename T>
struct is_same<T, T> {
    static constexpr bool value = true;
};

template <typename T>
void print_tile(T& t)
{
    t.print();

    printf("[");
    for (int i=0; i < T::M; ++i)
    {
        printf("%*s[", i>0, "");
        for (int j=0; j < T::N; ++j)
        {
            printf("%5.2f ", t.fwd(i*T::N + j));
        }

        if (i == T::M-1)
            printf("]]\n");
        else
            printf("]\n");
    }
}


template <typename Tile>
int size(Tile& t) { return Tile::M*Tile::N; }


template <typename T, int M_, int N_>
struct tile_load_t
{
    using Type = T;
    static constexpr int M = M_;
    static constexpr int N = N_;

    array_t<T> slice;

    tile_load_t() {}
    tile_load_t(array_t<T>& src, int x, int y)
    {
        assert(src.ndim == 2);

        // compute offsets into original array and store a view
        const int i = x*M;
        const int j = y*N;

        // slice into src
        if (src.data)
            slice.data = data_at_byte_offset(src, byte_offset(src, i, j));
        if (src.grad)
            slice.grad = grad_at_byte_offset(src, byte_offset(src, i, j));

        slice.shape[0] = M;
        slice.shape[1] = N;
        slice.strides[0] = src.strides[0];
        slice.strides[1] = src.strides[1];
        slice.ndim = 2;
    }

    Type fwd(int e) const
    {
        int i = e/N;
        int j = e%N;

        return index(slice, i, j);
    }

    void bwd(int e, const T& adj_ret) const
    {
        int i = e/N;
        int j = e%N;

        if (slice.grad)
            atomic_add(&index_grad(slice, i, j), adj_ret);
    }

    void print()
    {
        printf("tile_load_t<%d, %d>\n", M, N);
    }

};

template <typename Tile_>
struct tile_store_t
{
    using Tile = Tile_;
    using Type = typename Tile_::Type;
    static constexpr int M = Tile_::M;
    static constexpr int N = Tile_::N;

    array_t<Type> slice;
    Tile tile;

    tile_store_t() {}
    tile_store_t(array_t<Type>& dest, int x, int y, Tile& t) : tile(t)
    {
        assert(dest.ndim == 2);

        // compute offsets into original array and store a view
        const int i = x*M;
        const int j = y*N;

        // slice into dest
        if (dest.data)
            slice.data = data_at_byte_offset(dest, byte_offset(dest, i, j));
        if (dest.grad)
            slice.grad = grad_at_byte_offset(dest, byte_offset(dest, i, j));

        slice.shape[0] = M;
        slice.shape[1] = N;
        slice.strides[0] = dest.strides[0];
        slice.strides[1] = dest.strides[1];
        slice.ndim = 2;
    }

    void fwd(int e) const
    {
        int i = e/N;
        int j = e%N;

        index(slice, i, j) = tile.fwd(e);
    }

    void bwd(int e) const
    {
        int i = e/N;
        int j = e%N;

        // materialize gradient (runs entire graph backward), reading incoming grads from the dest
        if (slice.grad)
            tile.bwd(e, index_grad(slice, i, j));
    }

    void print()
    {
        printf("tile_load_t<%d, %d>-+", M, N);
        print(tile);
    }
};


template <typename T, int M_, int N_>
struct tile_constant_t
{
    using Type = T;
    static constexpr int M = M_;
    static constexpr int N = N_;

    T c;
    T* adj_c;

    tile_constant_t() {}
    tile_constant_t(const T& c, T& adj_c) : c(c), adj_c(&adj_c) {}

    Type fwd(int e)
    {
        return c;
    }

    void bwd(int e, const T& adj_ret)
    {
        *adj_c += adj_ret;
    }

    void print()
    {
        printf("tile_constant_t<%d, %d>-+", M, N);
        print(c);
        printf("\n");
    }
};

template <typename T, int M_, int N_>
struct tile_zeros_t
{
    using Type = T;
    static constexpr int M = M_;
    static constexpr int N = N_;

    tile_zeros_t() {}

    Type fwd(int e)
    {
        return Type(0.0);
    }

    void bwd(int e, const T& adj_ret) {}

    void print()
    {
        printf("tile_zeros_t<%d, %d>-+", M, N);
        print(c);
        printf("\n");
    }
};

template <typename T, int M_, int N_>
struct tile_ones_t
{
    using Type = T;
    static constexpr int M = M_;
    static constexpr int N = N_;

    tile_ones_t() {}

    Type fwd(int e)
    {
        return Type(1.0);
    }

    void bwd(int e, const T& adj_ret) {}

    void print()
    {
        printf("tile_ones_t<%d, %d>-+", M, N);
        print(c);
        printf("\n");
    }
};

template <typename Tile>
struct tile_unary_map_t
{
    using Type = typename Tile::Type;
    static constexpr int M = Tile::M;
    static constexpr int N = Tile::N;

    using FwdOp = Type(*)(Type);
    using AdjOp = void(*)(Type, Type&, Type&);

    Tile tile;
    
    FwdOp fwd_fn;
    AdjOp adj_fn;

    tile_unary_map_t() {}
    tile_unary_map_t(Tile& t, FwdOp fwd, AdjOp adj)  : tile(t), fwd_fn(fwd), adj_fn(adj) {}

    Type fwd(int e) const
    {
        return fwd_fn(tile.fwd(e));
    }

    void bwd(int e, Type adj_ret) const
    {
        Type adj_a = 0.0;

        adj_fn(tile.fwd(e), adj_a, adj_ret);

        tile.bwd(e, adj_a);
    }

    void print()
    {
        printf("tile_unary_map_t<%d, %d>-+", M, N);
        tile.print();
    }
};

template <typename TileA, typename TileB>
struct tile_binary_map_t
{
    static_assert(wp::is_same<typename TileA::Type, typename TileB::Type>::value, "Error");
    static_assert(TileA::M == TileB::M, "Error");
    static_assert(TileA::N == TileB::N, "Error");

    using Type = typename TileA::Type;
    static constexpr int M = TileA::M;
    static constexpr int N = TileA::N;

    using FwdOp = Type(*)(Type, Type);
    using AdjOp = void(*)(Type, Type, Type&, Type&, Type&);

    TileA tile_a;
    TileB tile_b;

    FwdOp fwd_fn;
    AdjOp adj_fn;

    tile_binary_map_t() {}
    tile_binary_map_t(const TileA& a, TileB& b, FwdOp fwd, AdjOp adj) : tile_a(a), tile_b(b), fwd_fn(fwd), adj_fn(adj) {}

    Type fwd(int e) const
    {
        Type a = tile_a.fwd(e);
        Type b = tile_b.fwd(e);

        return fwd_fn(a, b);
    }

    void bwd(int e, Type adj_ret) const
    {
        Type a = tile_a.fwd(e);
        Type b = tile_b.fwd(e);
 
        Type adj_a = 0.0;
        Type adj_b = 0.0;

        adj_fn(a, b, adj_a, adj_b, adj_ret);

        // recurse
        tile_a.bwd(e, adj_a);
        tile_b.bwd(e, adj_b);
    }

    void print()
    {
        printf("tile_binary_map_t<%d, %d>", M, N);
        printf("\n   -+");
        tile_a.print();
        printf("\n   -+");
        tile_b.print();
    }
};




//-----------------------------------------------------------------------------------------------------
// High level entry points for each op (correspond to one Warp builtin)

template <typename T, int M, int N>
tile_zeros_t<T, M, N> tile_zeros() { return tile_zeros_t<T, M, N>(); }

template <typename T, int M, int N>
tile_ones_t<T, M, N> tile_ones() { return tile_ones_t<T, M, N>(); }

// entry point for load
template <typename T, int M, int N>
tile_load_t<T, M, N> tile_load(array_t<T>& a, int x, int y)
{
    return tile_load_t<T, M, N>(a, x, y);
}

template <typename T, int M, int N>
void adj_tile_load(array_t<T>& a, int x, int y, array_t<T>& adj_a, int adj_x, int adj_y, const tile_load_t<T, M, N>& adj_ret)
{
    // nop
}


// entry point for store
template <typename T, typename Tile>
void tile_store(array_t<T>& dest, int x, int y, Tile& t)
{
    tile_store_t<Tile> op(dest, x, y, t);

    // execute op
    for (int i=threadIdx.x; i < size(op); i += blockDim.x)
        op.fwd(i);
}


template <typename T, typename Tile>
void adj_tile_store(array_t<T>& dest, int x, int y, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, Tile& adj_t)
{
    tile_store_t<Tile> op(dest, x, y, t);

    for (int i=threadIdx.x; i < size(op); i += blockDim.x)
        op.bwd(i);
}


// unary map
template <typename Tile>
tile_unary_map_t<Tile> tile_map_impl(typename tile_unary_map_t<Tile>::FwdOp fwd, typename tile_unary_map_t<Tile>::AdjOp adj, Tile& a)
{
    return tile_unary_map_t<Tile>(a, fwd, adj);
}

// binary map
template <typename TileA, typename TileB>
tile_binary_map_t<TileA, TileB> tile_map_impl(typename tile_binary_map_t<TileA, TileB>::FwdOp fwd, typename tile_binary_map_t<TileA, TileB>::AdjOp adj, TileA& a, TileB& b)
{
    return tile_binary_map_t<TileA, TileB>(a, b, fwd, adj);
}

// use macro to capture adjoint operator
#define tile_map(op, ...) tile_map_impl(op, adj_##op, __VA_ARGS__)
//#define tile_map(op, a) tile_map_impl(wp::##op, wp::##op, a)


// nop
void adj_tile_map_impl(void) {}
#define adj_tile_map(...) adj_tile_map_impl()

// use a macro to capture the adjoint var in the expression
#define tile_constant(T, M, N, var) tile_constant_t<T, M, N>(var, adj_##var)

} // namespace wp

#if 0

//-----------------------------------------------------
// c = a + b

// forward
auto var_0 = wp::tile_load<wp::float32,8,4>(var_A, x, y);
auto var_1 = wp::tile_load<wp::float32,8,4>(var_B, x, y);
auto var_2 = wp::tile_add(var_0, var_1);
wp::tile_store(var_C, x, y, var_2)

// reverse
wp::adj_store(var_C, x, y, var_2, adj_C, _, _, adj_2)
wp::adj_tile_add(var_0, var_1, adj_0, adj_1, adj_2)
wp::adj_tile_load(var_B, x, y, adj_B, _, _, adj_1);
wp::adj_tile_load(var_B, x, y, adj_B, _, _, adj_0);


//-----------------------------------------------------
// x = a[0]
// c = x*2.0 + x

// forward
auto var_0 = wp::tile_load<wp::float32,8,4>(var_A, x, y);
auto var_1 = wp::tile_mul(var_0, 2.0);
auto var_2 = wp::tile_add(var_0, var_1);
wp::tile_store(var_C, x, y, var_2)

struct adj_store_t
{
    adj_store_t()
    {

    }

    float bwd(int i, float adj_ret)
    {
        return array.grad[i];
    }
};

template <typename P>
struct adj_add_t
{
    adj_add_t(P& parent)
    {
        
    }

    float bwd(int i, float& adj_a, float& adj_b)
    {
        // evaluate parent
        float adj_ret = parent.bwd(i);

        adj_a += adj_ret;
        adj_b += adj_ret;
    }
};

template <typename T>
struct adj_tile
{
    adj_tile(T& parent)
    {

    }



};

void adj_tile_load(A, x, y, adj_A, adj_x, adj_y, adj_ret)
{
    for i in A(x,y):
        adj_A[i] += adj_ret(i);
}



// reverse
wp::adj_store(var_C, x, y, var_2, adj_C, _, _, adj_2)   // adj_2->adj_C
wp::adj_tile_add(var_0, var_1, adj_0, adj_1, adj_2)     // adj_0->adj_2->adj_C, adj_1->adj_2->adj_C
wp::adj_tile_mul(var_0, 2.0, adj_0, _, adj_1);          // adj_0->adj_1->adj_2->adj_C
wp::adj_tile_load(var_A, x, y, adj_A, _, _, adj_0);     // adj_A->adj_0->adj_1->adj_2->adj_C


#endif