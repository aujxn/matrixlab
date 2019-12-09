# GMRES for MTH343

This program is an implementation for GMRES with restart to solve sparse linear systems of equations
and a general purpose library for working with matrices, sparse and dense.

Features include matrix-matrix and matrix-vector operations, QR decomposition, linear least squares,
transposition, vector norms, input of matrix market format, and iterators over matrices.

The library is far from complete and fully documented, but it is a reasonable start and completely
usable.

The main binary is set up to run some tests on the effects of modifying the number of search directions
before restarting GMRES.

The implementation for GMRES can be found in src/matrix/gmres.rs

## Implementation

The GMRES function is broken down into a few smaller pieces. Those components are:
- Sparse matrix - dense vector multiplication
- Dense matrix - dense vector multiplication
- QR decomposition
- Backsolving least squares
- Normalizing dense vectors

With these helper functions the implementation of GMRES is quite straight forward.
This implementation is optimized to save old QR decompositions when the process is not restarted.

## Outcomes and Changing Max Search Directions

For well conditioned (randomly generated) matrices this implementation does very well with solving
sparse linear systems. It can solve systems over 1000x1000 in less than 100 iterations. With each
iterations requiring a matrix vector product and a linear least squares on an upper triangular matrix.
Increasing the number of search directions can reduce the number of iterations at the cost of memory
consumption and the operations at each iteration get exponentially more expensive. For a matrix of size
2000 it takes about 100 iterations with 5 search directions. Increasing this to 40 reduces this to about
65, a modest improvement. Increasing the search directions to above 100, though, still takes about 60
iterations and the operations are more expensive.

Systems can behave many different ways by changing the paramaters. Ill-conditioned matrices sometimes
do better with less restarts. I hypothesize this is because my implementation doesn't use modified
Gram-Schmidt for the orthogonalization process. As a result, more search directions is less numerically
stable. I would like to rewrite GMRES using the householder transformation to improve the numerical
stability to see if this has less stagnation issues on ill-conditioned systems.
