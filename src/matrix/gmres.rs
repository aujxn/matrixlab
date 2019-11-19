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
    /* Step one: allocation */

    // gmres only works for square matrices
    if !a.is_square() {
        return Err(Error::SizeMismatch);
    }

    // refers to the size of the square system we are solving
    let dimension = a.num_rows();

    /* Counters */

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

    /************
     * Matrices *
     ************/

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
        .enumerate()
        .map(|(i, _)| Vec::with_capacity(i + 1))
        .collect();

    /***********
     * Vectors *
     ***********/

    // current iterate's guess to the solution of Ax = b.
    let mut x: Vec<f64> = Vec::with_capacity(dimension);

    // intermediate vector for solving the least squares
    // beta = Q^t * r
    let mut beta: Vec<f64> = Vec::with_capacity(max_search_directions);

    // solution to the least squares to compute next iterate's x and residual
    let mut alpha: Vec<f64> = vec![0.0; max_search_directions];

    // current iterate's residual (error) calculated: b - Ax.
    let mut residual: Vec<f64> = Vec::with_capacity(dimension);

    /* Step two: initialization */

    // first guess is the 0 vector
    x = (0..dimension).map(|_| 0.0).collect();
    //println!("initial search vector: {:?}", x);

    // first residual is just b, because b - Ax is the same as b - 0 (Ax = 0)
    residual = b.get_data().iter().map(|&x| x).collect();
    let mut residual_norm = residual.iter().fold(0.0, |acc, x| acc + x * x).sqrt();
    //println!("initial residual: {:?}", residual);
    //println!("initial residual norm: {:?}", residual_norm);

    /* Step three: iterate */
    loop {
        //println!("m: {:?}", m);
        // reset the workspaces when max search directions is hit (or starting first iteration)
        if m == 1 {
            // scale the residual so it is normalized. This is our first column of P
            workspace_p[0] = residual.iter().map(|x| x * (1.0 / residual_norm)).collect();
            //println!("first search direction: {:?}", workspace_p[0]);

            // perform the matrix vector multiplication Ap_0 to get first B column
            workspace_b[0] = a
                .row_iter()
                .map(|(cols, data)| {
                    data.iter()
                        .zip(cols.iter())
                        .fold(0.0, |acc, (val, &col)| acc + val * workspace_p[0][col])
                })
                .collect();
            //println!("first B column: {:?}", workspace_b[0]);

            // first orthonormal decomposition column of B by normalizing b_zero
            let b_zero_norm = workspace_b[0].iter().fold(0.0, |acc, x| acc + x * x).sqrt();
            workspace_q[0] = workspace_b[0]
                .iter()
                .map(|x| x * (1.0 / b_zero_norm))
                .collect();
            //println!("first column of Q: {:?}", workspace_q[0]);
            //println!("first B column norm: {:?}", b_zero_norm);

            // upper triangular R is a matrix with a single value
            workspace_r[0].clear();
            workspace_r[0].push(b_zero_norm);
            //println!("upper triangle with one element (same as last output) {:?}", workspace_r[0]);
        }

        // compute beta vector in order to solve least squares
        beta = workspace_q
            .iter()
            .take(m)
            .map(|row| {
                row.iter()
                    .zip(residual.iter())
                    .fold(0.0, |acc, (q_val, r_val)| acc + q_val * r_val)
            })
            .collect();
        //println!("beta: {:?}", beta);

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
        //println!("alpha: {:?}", alpha);

        // compute the next iterate
        /*
        x = x.iter()
            .enumerate()
            .map(|(row, x_val)| {
                x_val + (0..m).fold(0.0, |acc, col| acc + workspace_p[col][row] * alpha[col])})
            .collect();
        */
        x = (0..dimension)
            .map(|row| {
                (0..m)
                    .map(|col| workspace_p[col][row])
                    .zip(alpha.iter())
                    .fold(0.0, |acc, (p_val, alpha_val)| acc + p_val * alpha_val)
            })
            .zip(x.iter())
            .map(|(p_alpha, x_val)| p_alpha + x_val)
            .collect();
        //println!("new iterate: {:?}", x);

        // compute the next residual
        residual = (0..dimension)
            .map(|row| {
                (0..m)
                    .map(|col| workspace_b[col][row])
                    .zip(alpha.iter())
                    .fold(0.0, |acc, (b_val, alpha_val)| acc + b_val * alpha_val)
            })
            .zip(residual.iter())
            .map(|(b_alpha, r_val)| r_val - b_alpha)
            .collect();
        residual_norm = residual.iter().fold(0.0, |acc, x| acc + x * x).sqrt();
        /*
        if new_residual_norm > residual_norm {
            panic!("residual increased");
        } else {
            residual_norm = new_residual_norm;
        }
        */
        //println!("new residual: {:?}", residual);
        //println!("new residual norm: {:?}", residual_norm);

        if residual_norm < tolerance {
            println!("Iterations: {:?}", iteration);
            return Ok(DenseVec::new(x));
        }

        iteration += 1;
        if iteration % 100 == 0 {
            println! {"{:?}", residual_norm};
        }
        //println!("\niteration: {:?}", iteration);

        if iteration == max_iter {
            return Err(Error::ExceededIterations(x.clone()));
        }

        if m < max_search_directions {
            // compute next search direction

            // calculate inner products of P columns and residual vector
            workspace_r[m] = workspace_p
                .iter()
                .take(m)
                .map(|p_col| {
                    p_col
                        .iter()
                        .zip(residual.iter())
                        .fold(0.0, |acc, (p_val, r_val)| acc + p_val * r_val)
                })
                .collect();
            // calculate the next search vector
            workspace_p[m] = residual
                .iter()
                .enumerate()
                .map(|(row, r_val)| {
                    let sum = workspace_p
                        .iter()
                        .take(m)
                        .enumerate()
                        .fold(0.0, |sum, (col, p_col)| {
                            sum + workspace_r[m][col] * p_col[row]
                        });
                    r_val - sum
                })
                .collect();
            //println!("new column of P (pre - normalized): {:?}", workspace_p[m]);
            // normalize the search vector
            let norm = workspace_p[m].iter().fold(0.0, |acc, x| acc + x * x).sqrt();
            //println!("norm of new column of P: {:?}", norm);
            workspace_p[m] = workspace_p[m]
                .iter()
                .map(|p_value| p_value * 1.0 / norm)
                .collect();
            //println!("new column of P (normalized): {:?}", workspace_p[m]);

            // add next krylov vector to B
            workspace_b[m] = a
                .row_iter()
                .map(|(cols, data)| {
                    data.iter()
                        .zip(cols.iter())
                        .fold(0.0, |acc, (val, &col)| acc + val * workspace_p[m][col])
                })
                .collect();
            //println!("new column of B: {:?}", workspace_b[m]);
            let norm = workspace_b[m].iter().fold(0.0, |acc, x| acc + x * x).sqrt();
            //println!("norm of new column of B: {:?}", norm);

            // calculate inner products of Q columns and new B column to make new R column
            workspace_r[m] = workspace_q
                .iter()
                .take(m)
                .map(|q_col| {
                    q_col
                        .iter()
                        .zip(workspace_b[m].iter())
                        .fold(0.0, |acc, (q_val, b_val)| acc + q_val * b_val)
                })
                .collect();
            workspace_r[m].push(norm);
            //println!("new column of R: {:?}", workspace_r[m]);

            // calculate next orthonormal vector to Q column from new R column
            workspace_q[m] = workspace_b[m]
                .iter()
                .enumerate()
                .map(|(row, b_val)| {
                    let sum = workspace_q
                        .iter()
                        .take(m)
                        .enumerate()
                        .fold(0.0, |sum, (col, q_col)| {
                            sum + workspace_r[m][col] * q_col[row]
                        });
                    b_val - sum
                })
                .collect();
            let norm = workspace_q[m].iter().fold(0.0, |acc, x| acc + x * x).sqrt();
            workspace_q[m] = workspace_q[m]
                .iter()
                .map(|q_value| q_value * 1.0 / norm)
                .collect();
            //println!("new column of Q: {:?}", workspace_q[m]);

            // update search direction counter
            m += 1;
        } else {
            // reset the search direction counter
            m = 1;
            //println!("restarting....");
        }
    }
}
