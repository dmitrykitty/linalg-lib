# API reference

Containers and operations of the library. The [README](../README.md) has the overview, performance
results, and build instructions; [performance.md](performance.md) documents the measurements.

## Containers

### Matrix (`DynamicMatrix<T>`)

- contiguous row-major `float` or `double` storage through `DynamicMatrix<T>`;
- `Matrix` remains an alias for `DynamicMatrix<double>`;
- default, dimension, fill, flat-sequence, and nested initializer-list construction;
- copy and move ownership semantics;
- `rows()`, `cols()`, `size()`, `empty()`, and contiguous `data()` access;
- unchecked `operator()(row, col)` and checked `at(row, col)` access;
- in-place matrix addition and subtraction;
- in-place scalar addition, subtraction, and multiplication;
- value-returning addition, subtraction, and scalar multiplication;
- matrix-vector multiplication;
- matrix-matrix multiplication through a cache-blocked, packed GEMM: AVX2/FMA micro-kernel chosen
  at run time (portable fallback), multithreaded above 2 x 128^3 FLOP
  ([how it works](../README.md#how-the-gemm-works));
- adding a vector to every row or every column;
- constructing a matrix by repeating a vector as rows or columns;
- transpose and trace;
- one, Frobenius, and infinity norms selected with `MatrixNorm`.

For a float matrix, use `linalg::DynamicMatrix<float>`:

```cpp
linalg::DynamicMatrix<float> small{{1.0f, 2.0f}, {3.0f, 4.0f}};
small *= 2.0f;
```

Construction, access, arithmetic, transpose, matrix-vector multiplication, and matrix-matrix
multiplication work for both scalar types. Trace, norms, broadcasting, and repeating vectors
currently accept the double-precision aliases; their float versions are a later migration step.

### Static matrix (`StaticMatrix<T, Rows, Cols>`)

`StaticMatrix<T, Rows, Cols>` stores a fixed number of `float` or `double` elements directly
inside the object using `std::array`. Its dimensions are part of the type:

```cpp
linalg::StaticMatrix<double, 2, 3> matrix{{1.0, 2.0, 3.0}, {4.0}};
matrix.at(1, 2) = 6.0;

constexpr linalg::StaticMatrix<float, 2, 2> identity{{1.0f, 0.0f}, {0.0f, 1.0f}};
static_assert(identity(1, 1) == 1.0f);
```

Static matrices provide construction, ownership, and the same core access API as dynamic
matrices: `rows()`, `cols()`, `size()`, `empty()`, `data()`, `operator()`, and `at()`. Transpose,
matrix-matrix multiplication, and matrix-vector multiplication support both scalar types and
return fixed-shape results. Elementwise arithmetic, norms, trace, and dynamic/static conversions
are not provided for static matrices yet.

- Default construction fills every element with zero.
- Parentheses fill every element: `StaticMatrix<double, 2, 3>(4.0)`.
- `StaticMatrix<double, 2, 3>::from_values({4.0})` initializes only the first element to 4.0.
  The named flat-list factory avoids ambiguity with nested lists such as `{{1.0}, {2.0}}`.
- `from_values()` and explicit `std::span<const T>` construction copy row-major values and
  zero-pad the rest.
- Nested lists zero-pad short rows and missing rows. Too many rows, too many columns in a row,
  or too many flat values throw `std::invalid_argument`; the fixed shape never expands.
- Zero dimensions are supported. Shapes whose storage-size calculation would overflow are
  rejected at compile time.
- Copies own independent storage. Compiler-generated moves operate on the inline array;
  both objects keep their fixed shape, and moving `float`/`double` elements copies their values.

Construction, access, transpose, and multiplication support `constexpr`, so valid small examples
can be evaluated at compile time. Ordinary calls still execute at runtime. Inline storage avoids
a separate heap allocation,
but the object's location determines whether that storage is on the stack, heap, or elsewhere.
Prefer dynamic matrices for large or runtime-sized shapes.

Both owning types share initialization and indexing helpers in `include/linalg/internal/`;
storage and dimension validation remain specific to each type. This keeps the implementation
small without introducing inheritance or a storage-policy framework.

### Vector (`DynamicVector<T>`)

- contiguous owning `float` or `double` storage through `DynamicVector<T>`;
- `Vector` remains an alias for `DynamicVector<double>`;
- default, size, fill, initializer-list, and `std::span` construction;
- copy and move ownership semantics;
- `size()`, `empty()`, and contiguous `data()` access;
- unchecked `operator[]` and checked `at()` access;
- in-place vector addition and subtraction;
- in-place scalar addition, subtraction, and multiplication;
- value-returning addition, subtraction, and scalar multiplication;
- L2 normalization with validation for zero and non-finite vectors;
- L1, L2, and infinity norms selected with `VectorNorm`.

### Static vector (`StaticVector<T, Size>`)

`StaticVector<T, Size>` owns a fixed-length `std::array` of `float` or `double` elements.
It provides construction, copy/move semantics, `size()`, `empty()`, `data()`, `operator[]`,
and checked `at()` access:

```cpp
linalg::StaticVector<double, 3> values{1.0, 2.0}; // {1.0, 2.0, 0.0}
linalg::StaticVector<double, 3> filled(4.0); // {4.0, 4.0, 4.0}
values.at(2) = 3.0;

constexpr linalg::StaticVector<float, 3> axis{1.0f, 0.0f, 0.0f};
static_assert(axis.size() == 3 && axis[0] == 1.0f);
```

The template specifies the length; parentheses fill every element rather than supplying a
runtime size. Braces contain values and zero-pad any remaining elements. Explicit
`std::span<const T>` construction likewise copies values and pads missing elements. Excess
input throws `std::invalid_argument`; `at()` throws `std::out_of_range` for invalid indices.
Unchecked access requires a valid index and asserts that precondition in Debug builds.

Zero-length vectors are supported, and lengths whose byte-size calculation would overflow
are rejected at compile time. Inline storage has no separate heap allocation. Copies and
moves use the compiler-generated `std::array` behavior: each object retains its fixed length
and independent storage, and moving `float`/`double` values copies them.

Construction and access support `constexpr`. Multiplying a matching static matrix by a static
vector returns a static vector and also supports constant evaluation. Static vector arithmetic,
norms, normalization, and dynamic/static conversions remain later steps.

Matrices and static vectors reuse input validation and copying helpers in
`include/linalg/internal/storage.hpp`.
Dynamic and static vectors also share checked and unchecked access helpers; their owning storage
remains separate.

## Construction, scalar types, and errors

For `Vector`, braces contain values while parentheses specify a size:

```cpp
linalg::Vector one_value{3.0}; // one element containing 3.0
linalg::Vector three_zeroes(3); // three zero-initialized elements
```

`DynamicVector<T>` likewise supports `float` and `double`. Its constructors, element access,
and arithmetic work for both types:

```cpp
linalg::DynamicVector<float> samples{1.0f, 2.0f, 3.0f};
samples *= 2.0f;
```

Existing `linalg::Vector` code still uses `double`. Vector norms, normalization, and broadcast
functions currently take the double-precision aliases; their float versions are a later migration
step. Matrix-vector multiplication supports both scalar types. Operations between different
scalar types are not provided yet; convert elements explicitly when needed.

Vector addition and subtraction require equal sizes. Matrix addition and subtraction require equal
shapes. Broadcast operations require a vector length matching the relevant matrix dimension.
Trace requires a square, non-empty matrix. Invalid dimensions throw `std::invalid_argument`;
checked element access throws `std::out_of_range`.

## Shared transpose and multiplication

The public operations select the result container; each calculation lives in one shared kernel
under `include/linalg/internal/kernels.hpp`. These internal helpers overwrite a preallocated,
non-aliasing result and are not public API. No inheritance or duplicated dynamic/static loops are
needed.

```cpp
linalg::DynamicMatrix<float> dynamic{{1.0f, 2.0f}, {3.0f, 4.0f}};
auto dynamic_product = linalg::multiply(dynamic, dynamic); // DynamicMatrix<float>

constexpr linalg::StaticMatrix<double, 2, 3> fixed{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
constexpr linalg::StaticVector<double, 3> vector{7.0, 8.0, 9.0};
constexpr auto transposed = linalg::transpose(fixed); // StaticMatrix<double, 3, 2>
constexpr auto applied = linalg::multiply(fixed, vector); // StaticVector<double, 2>
constexpr auto product = linalg::multiply(fixed, transposed); // StaticMatrix<double, 2, 2>
static_assert(applied[0] == 50.0 && product(0, 0) == 14.0);
```

Dynamic multiplication checks dimensions at runtime and throws `std::invalid_argument` on a
mismatch. Incompatible static multiplication shapes have no matching overload and are rejected
at compile time. Public multiplication currently requires the same scalar type and either two
dynamic operands or two static operands; mixed ownership is not exposed yet.

The library uses indexed column-row transpose (contiguous destination writes). Static matrices
multiply with the `constexpr` `ikj` kernel (contiguous reads of the right matrix and updates of the
result), which suits small fixed shapes and constant evaluation. Dynamic matrices multiply with the
blocked GEMM.

The blocked GEMM behind dynamic matrix multiplication is described in the
[README](../README.md#how-the-gemm-works) and measured in [performance.md](performance.md).
