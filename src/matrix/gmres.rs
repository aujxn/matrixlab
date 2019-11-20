/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use crate::error::Error;
use crate::matrix::sparse::SparseMatrix;
use crate::vector::dense::DenseVec;
use rayon::prelude::*;

// I didn't use this library's abstractions for the data structures in this method.
// The reason for this is that this is a highly iterative process, and I wanted
// complete control of the allocations that are made. This makes this function
// slightly more convoluted but much more efficient.
// To compensate for this, I have written very detailed comments.
/// Solves a linear system using the generalized minimal residual method.
/// Only implemented for SparseMatrix<f64>
///
/// # Example
///
/// ```
/// use matrixlab::MatrixElement;
/// use matrixlab::matrix::sparse::SparseMatrix;
/// use matrixlab::vector::dense::DenseVec;
/// use matrixlab::matrix::gmres::gmres;
///
/// let elements = vec![MatrixElement(0, 0, 2f64), MatrixElement(1, 1, 2f64), MatrixElement(0, 1, 1.0)];
/// let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();
///
/// let result = gmres(mat, DenseVec::new(vec![3.0, 2.0]), 1.0 / 1000000.0, 1000, 2)
///     .unwrap();
///     assert_eq!(result, DenseVec::new(vec![1.0, 1.0]));
/// ```
pub fn gmres(
    a: SparseMatrix<f64>,         // The system to solve. Referred to as A below.
    b: DenseVec<f64>,             // The vector to solve Ax = b for x.
    tolerance: f64,               // The acceptable level of residual. Calculated b - Ax.
    max_iter: usize,              // Maximum number of times to guess the solution of x.
    max_search_directions: usize, // The number of search vectors to consider before restarting.
) -> Result<DenseVec<f64>, Error> {
    // gmres only works for square matrices
    if !a.is_square() {
        return Err(Error::SizeMismatch);
    }

    // refers to the size of the square system we are solving
    let dimension = a.num_rows();

    // the current index of search vectors being considered
    // m is the number of columns in the workspaces detailed
    // below (B, P, Q, and both dimensions of R) as well as
    // the length of the vectors alpha and beta.
    let mut m = 1;

    // how many times of the algorithm loop has elapsed
    let mut iteration = 0;

    /*********************************************************************
     *                            Workspaces                             *
     *********************************************************************/

    // Indexing these workspaces is reverse from most of the lib. Because
    // we will be adding columns to these matrices, they are indexed by
    // column on the outer dimension and then row on the inner.

    // workspace_b (B):
    // This is essentially the Krylov subspace for matrix A. This is the
    // matrix used in the least squares problem needed to find the next
    // search direction. It can have up to the max number of search
    // directions of vectors with lengths equal to either dimension of A.
    let mut workspace_b: Vec<Vec<f64>> = (0..max_search_directions)
        .map(|_| Vec::with_capacity(dimension))
        .collect();

    // workspace_p (P):
    // These are the vectors used to calculate the vectors in B.
    // It is the same size as B. The columns of this matrix are
    // basically the search directions of the process. The initial column
    // can be the zero vector or a random vector.
    let mut workspace_p: Vec<Vec<f64>> = (0..max_search_directions)
        .map(|_| Vec::with_capacity(dimension))
        .collect();

    // workspace_q (Q):
    // This is the Q componant of the QR decomposition of B.
    // This makes solving the least squares problems very fast.
    // The columns of this matrix are all orthonormal.
    // It is also the same dimension as P and B.
    let mut workspace_q: Vec<Vec<f64>> = (0..max_search_directions)
        .map(|_| Vec::with_capacity(dimension))
        .collect();

    // workspace_r (R):
    // The R componant of the QR decomposition of B.
    // This is an upper triangular matrix - is the second key
    // part to solving the least squares problems efficiently.
    // Only data above the diagonal is stored.
    let mut workspace_r: Vec<Vec<f64>> = (0..max_search_directions)
        .map(|i| Vec::with_capacity(i + 1))
        .collect();

    // current iterate's guess to the solution of Ax = b.
    // first guess is the 0 vector
    let mut x: Vec<f64> = vec![0.0; dimension];
    //println!("initial search vector: {:?}", x);

    // intermediate vector for solving the least squares
    // beta = Q^t * r
    let mut beta: Vec<f64>;

    // solution to the least squares to compute next iterate's x and residual
    let mut alpha: Vec<f64> = vec![0.0; max_search_directions];

    // current iterate's residual (error) calculated: b - Ax.
    // first residual is just b, because b - Ax is the same as b - 0 (Ax = 0)
    let mut residual: Vec<f64> = b.get_data().clone();
    let mut residual_norm = norm(&residual);
    //println!("initial residual: {:?}", residual);
    //println!("initial residual norm: {:?}", residual_norm);

    /* Step three: iterate */
    loop {
        // reset the workspaces when max search directions is hit (or starting first iteration)
        if m == 1 {
            // scale the residual so it is normalized. This is our first column of P
            workspace_p[0] = normalize(&residual, residual_norm);

            // perform the matrix vector multiplication Ap_0 to get first B column
            workspace_b[0] = sparse_matrix_dot_vec(&a, &workspace_p[0]);

            // first orthonormal decomposition column of B by normalizing b_zero
            let b_zero_norm = norm(&workspace_b[0]);
            workspace_q[0] = normalize(&workspace_b[0], b_zero_norm);

            // upper triangular R is a matrix with a single value
            workspace_r[0].clear();
            workspace_r[0].push(b_zero_norm);
        }

        // compute beta vector in order to solve least squares
        // with Q_transpose * residual
        beta = dense_matrix_transpose_dot_vec(&workspace_q, m, &residual);

        // backsolve least squares
        for row in (0..m).rev() {
            let inner = workspace_r
                .iter()
                .skip(row + 1)
                .take(m - (row + 1))
                .map(|col| col[row])
                .zip(alpha.iter().skip(row + 1))
                .fold(0.0, |acc, (r_val, alpha_val)| acc + r_val * alpha_val);
            alpha[row] = (beta[row] - inner) / workspace_r[row][row];
        }

        // compute the next iterate
        x = dense_matrix_dot_vec(&workspace_p, m, &alpha)
            .iter()
            .zip(x.iter())
            .map(|(p_alpha, x_val)| p_alpha + x_val)
            .collect();

        // compute the new residual
        residual = dense_matrix_dot_vec(&workspace_b, m, &alpha)
            .iter()
            .zip(residual.iter())
            .map(|(b_alpha, r_val)| r_val - b_alpha)
            .collect();
        residual_norm = norm(&residual);

        // when residual norm is less than tolerance, x is solution
        if residual_norm < tolerance {
            println!("Iterations: {:?}", iteration);
            return Ok(DenseVec::new(x));
        }

        iteration += 1;

        if iteration % 100 == 0 {
            println! {"{:?}", residual_norm};
        }

        if iteration == max_iter {
            return Err(Error::ExceededIterations(x.clone()));
        }

        // do not restart gmres
        if m < max_search_directions {
            // compute next search direction

            // calculate inner products of P columns and residual vector
            workspace_r[m] = workspace_p
                .iter()
                .take(m)
                .map(|p_col| inner(&p_col, &residual))
                .collect();
            // calculate the next search vector
            workspace_p[m] = orthogonal_to(&workspace_p, m, &residual, &workspace_r[m]);

            // normalize the search vector
            let p_norm = norm(&workspace_p[m]);
            workspace_p[m] = normalize(&workspace_p[m], p_norm);

            // add next krylov vector to B
            workspace_b[m] = sparse_matrix_dot_vec(&a, &workspace_p[m]);

            //println!("new column of B: {:?}", workspace_b[m]);
            let b_norm = norm(&workspace_b[m]);

            // calculate inner products of Q columns and new B column to make new R column
            workspace_r[m] = workspace_q
                .iter()
                .take(m)
                .map(|q_col| inner(&q_col, &workspace_b[m]))
                .collect();
            workspace_r[m].push(b_norm);

            // calculate next orthonormal vector to Q column from new R column
            workspace_q[m] = orthogonal_to(&workspace_q, m, &workspace_b[m], &workspace_r[m]);
            let q_norm = norm(&workspace_q[m]);
            workspace_q[m] = normalize(&workspace_q[m], q_norm);

            m += 1;
        } else {
            // restart gmres by resetting the search direction counter
            m = 1;
        }
    }
}

fn norm(vector: &Vec<f64>) -> f64 {
    vector.iter().fold(0.0, |acc, x| acc + x * x).sqrt()
}

fn normalize(vector: &Vec<f64>, norm: f64) -> Vec<f64> {
    vector.iter().map(|x| x * (1.0 / norm)).collect()
}

fn inner(vector1: &Vec<f64>, vector2: &Vec<f64>) -> f64 {
    vector1
        .iter()
        .zip(vector2.iter())
        .fold(0.0, |inner, (val_1, val_2)| inner + val_1 * val_2)
}

fn orthogonal_to(
    orthonormal_matrix: &Vec<Vec<f64>>,
    cols: usize,
    vector: &Vec<f64>,
    coefficient: &Vec<f64>,
) -> Vec<f64> {
    vector
        .iter()
        .enumerate()
        .map(|(row, vec_val)| {
            let sum = orthonormal_matrix
                .iter()
                .take(cols)
                .enumerate()
                .fold(0.0, |sum, (col, mat_col)| {
                    sum + coefficient[col] * mat_col[row]
                });
            vec_val - sum
        })
        .collect()
}

fn dense_matrix_dot_vec(matrix: &Vec<Vec<f64>>, cols: usize, vector: &Vec<f64>) -> Vec<f64> {
    let dimension = matrix[0].len();
    (0..dimension)
        .map(|row| {
            (0..cols)
                .map(|col| matrix[col][row])
                .zip(vector.iter())
                .fold(0.0, |acc, (mat_val, vec_val)| acc + mat_val * vec_val)
        })
        .collect()
}

fn dense_matrix_transpose_dot_vec(
    matrix: &Vec<Vec<f64>>,
    rows: usize,
    vector: &Vec<f64>,
) -> Vec<f64> {
    matrix
        .iter()
        .take(rows)
        .map(|row| {
            row.iter()
                .zip(vector.iter())
                .fold(0.0, |acc, (mat_val, vec_val)| acc + mat_val * vec_val)
        })
        .collect()
}

fn sparse_matrix_dot_vec(matrix: &SparseMatrix<f64>, vector: &Vec<f64>) -> Vec<f64> {
    matrix
        .row_iter()
        .map(|(cols, data)| {
            data.iter()
                .zip(cols.iter())
                .fold(0.0, |acc, (val, &col)| acc + val * vector[col])
        })
        .collect()
}
