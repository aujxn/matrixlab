/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use crate::error::Error;
use crate::Element;
use crate::MatrixElement;
use crate::vector::dense::DenseVec;
use crate::vector::sparse::SparseVec;
use rayon::prelude::*;
use std::ops::{AddAssign, Add, Mul, Sub};
use std::iter::FromIterator;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::ShapeError;
use ndarray::Array;
use ndarray::Ix2;
use ndarray::ScalarOperand;

#[derive(PartialEq, Debug)]
/// A dense matrix is a vector of dense vectors.
/// Each dense vector is a column of the matrix.
pub struct DenseMatrix<A> {
    data: Array<A, Ix2>,
}

impl<A: Element + Default + AddAssign> DenseMatrix<A> {
    /// Creates a dense matrix from a set of elements. If multiple non-zero
    /// elements are found at the same locations the sum is taken.
    pub fn from_triplets(num_rows: usize, num_columns: usize, elements: Vec<MatrixElement<A>>) -> Result<DenseMatrix<A>, Error> {
        let mut data = Array2::default((num_rows, num_columns));

        for MatrixElement(i, j, val) in elements {
            if let Some(element) = data.get_mut((i, j)) {
                *element += val;
            } else {
                return Err(Error::ElementOutOfBounds)
            }
        }

        Ok(DenseMatrix { data })
    }
}

impl<A: Element> DenseMatrix<A> {
    /// Creates a matrix from a vector of columns.
    pub fn from_columns(num_rows: usize, num_columns: usize, columns: Vec<Vec<A>>) -> Result<DenseMatrix<A>, ShapeError> {
        let mut iter = columns.iter().flatten().map(|&x| x);
        let mut data: Array<A, Ix2> = Array::from_iter(iter).into_shape((num_columns, num_rows))?;
        data.swap_axes(0, 1);
        Ok(DenseMatrix { data })
    }

    /// Creates a matrix from a vector of rows.
    pub fn from_rows(num_rows: usize, num_columns: usize, rows: Vec<Vec<A>>) -> Result<DenseMatrix<A>, ShapeError> {
        let mut iter = rows.iter().flatten().map(|&x| x);
        let mut data: Array<A, Ix2> = Array::from_iter(iter).into_shape((num_rows, num_columns))?;
        Ok(DenseMatrix { data })
    }

    /// Creates a matrix from a 2 dimensional array created with the
    /// ndarray crate. Dimension 0 is referred to as rows or i and 1
    /// is referred to as columns or j.
    pub fn new(data: Array2<A>) -> DenseMatrix<A> {
        DenseMatrix { data }
    }

    /// Replaces an entire column of the matrix
    pub fn replace_column(&mut self, column: Vec<A>, index: usize) {
    }

    /// Replaces an entire row of the matrix
    pub fn replace_row(&mut self, row: Vec<A>, index: usize) {
    }

    /// Returns a mutable reference to the array of data
    pub fn get_mut_data(&mut self) -> &mut Array2<A> {
        &mut self.data
    }

    /// Returns a immutable reference to the array of data
    pub fn get_data(&mut self) -> &Array2<A> {
        &self.data
    }
}

//Maybe todo: should this be copy?
impl<A: Element> DenseMatrix<A> {
    /// Returns the transpose of the matrix
    pub fn transpose(&self) -> DenseMatrix<A> {
        let data = self.data.clone().reversed_axes();
        DenseMatrix::new(data)
    }

    /// Transpose the matrix in place
    pub fn transpose_self(&mut self) {
        self.data.swap_axes(0, 1);
    }

    /// Returns the number of rows in the matrix
    pub fn num_rows(&self) -> usize {
        self.data.nrows()
    }

    /// Returns the number of columns in the matrix
    pub fn num_columns(&self) -> usize {
        self.data.ncols()
    }
}

impl<A: Element + ScalarOperand + Mul<Output = A>> DenseMatrix<A> {
    /// Returns a new matrix with every element scaled by some factor
    pub fn scale(&self, factor: &A) -> DenseMatrix<A> {
        DenseMatrix::new(&self.data * *factor)
    }

    /// Modifies the matrix by scaling every element
    pub fn scale_mut(&mut self, factor: &A) {
        self.data.mapv_inplace(|x| x * *factor);
    }
}

//TODO: Make this generic?
impl<A: Element + Mul<Output = A> + Add<Output = A> + Sub<Output = A> + Default> DenseMatrix<A> {
    /// Multiply a dense matrix by a dense vector
    pub fn dense_vec_mul(&self, other: &DenseVec<A>) -> DenseVec<A> {
        let result = self
            .data
            .axis_iter(Axis(0))
            .map(|row| {
                row
                    .iter()
                    .zip(other.iter())
                    .fold(A::default(), |acc, (&i, &j)| acc + i * j)
            })
            .collect();

        DenseVec::new(result)
    }

    /* TODO??:
    pub fn safe_dense_vec_mul(&self, other: &DenseVec<A>) -> DenseVec<A> {
    }

    pub fn sparse_vec_mul(&self, other: &SparseVec<A>) -> DenseVec<A> {
    }

    pub fn safe_sparse_vec_mul(&self, other: &SparseVec<A>) -> DenseVec<A> {
    }
    */

    /// Checked composition of two dense matrices. Returns a SizeMismatch error
    /// if dimensions are invalid
    pub fn safe_dense_mat_mul(&self, other: &DenseMatrix<A>) -> Result<DenseMatrix<A>, Error> {
        if self.num_columns() != other.num_rows() {
            return Err(Error::SizeMismatch);
        }

        // TODO: not parallel anymore >:(
        let new = Array::from_iter(
            (0..self.data.nrows())
            .into_iter()
            .map(|left_row| {
                (0..other.data.ncols())
                    .into_iter()
                    .map(move |right_column| {
                        self.data.row(left_row)         // gets a view of the left matrix's row
                            .iter()
                            .zip(other.data.column(right_column).iter())    // and zips it with right column
                            .fold(A::default(), |inner, (&i, &j)| inner + i * j)    //inner product
                    })
            }).flatten()
        ).into_shape((self.num_rows(), other.num_columns()))?;

        Ok(DenseMatrix::new(new))
    }
}

impl DenseMatrix<f64> {
    /// This takes an upper triangular matrix, and solves it to
    /// equal b
    pub fn backsolve(&self, b: &DenseVec<f64>) -> DenseVec<f64> {
        // Start off with a copy of b, to modify to create our solutions
        let mut solutions: Vec<f64> = b.get_data().clone();
        // Start with the last column
        for (i, column) in self.data.axis_iter(Axis(1)).rev().enumerate() {
            //Normalize our last element
            let last_element = b.len() - 1 - i;
            solutions[last_element] /= column[last_element];
            //And skip i elements because they're all zero
            //But we have to reverse the list first
            for (j, element) in column.iter().rev().skip(1 + i).enumerate() {
                //Move up b as we iterate
                let last_element = b.len() - 1 - i;
                //And move up b as we go up each column
                //This probably won't overflow
                //TODO ^ figure out if this is exploitable
                let current_element = last_element - 1 - j;
                solutions[current_element] -= solutions[last_element] * element;
            }
        }
        DenseVec::new(solutions)
    }

    /// This solves for B*y = r
    pub fn least_squares(&self, r: &DenseVec<f64>) -> Result<DenseVec<f64>, Error> {
        //Solve for Q, for our QR factorization
        let q = self.factor_q();
        let q_transpose = q.transpose();
        let rhs = q_transpose.dense_vec_mul(&r);
        let r = q_transpose
            .safe_dense_mat_mul(self)
            .expect("Error in least squares, multiplication failed");;

        //Now solve for Ra = rhs, then return a
        Ok(r.backsolve(&rhs))
    }

    /// Takes all the columns in a dense matrix and finds a corresponding
    /// set of vectors that are all ortholinear and span the same space.
    /// This method is known as Gram-Schmidt process.
    pub fn factor_q(&self) -> DenseMatrix<f64> {
        let mut q_vectors: Vec<DenseVec<f64>> = Vec::with_capacity(self.num_columns());
        for column in self.data.axis_iter(Axis(1)) {
            //TODO: change from vec to workspace array to avoid repeated allocs
            //just in case the compiler isn't doing it already
            let mut maybe_q = DenseVec::new(column.iter().map(|&x| x).collect());
            for orthogonal in q_vectors.iter() {
                // Is this the numerical stable MGS or normal gram schmidt? TODO: look closer later
                let c = column.iter().zip(orthogonal.iter()).fold(0.0, |acc, (&i, &j)| acc + i * j);
                // subtract c_n * q_n
                maybe_q.sub_mut(&orthogonal.scale(c));
            }
            let q = maybe_q.normalize();
            q_vectors.push(q);
        }

        // TODO: Since we are likel to transpose this matrix for solving the
        // decomposed system should we just build it transposed? Actually,
        // with the ndarray transposition of dense matrices is free so this
        // might not be a big deal.
        let matrix: Array<f64, Ix2> = Array::from_iter(
            q_vectors
            .iter()
            .map(|cols| {
                cols
                .get_data()
                .iter()
                .map(|&x| x)
            }).flatten()
            )
            .into_shape((self.num_rows(), self.num_columns())).unwrap();

        DenseMatrix::new(matrix)
    }

    /// Returns a new vector, orthogonal to all vectors currently in the
    /// matrix and to the other vector
    pub fn orthogonal_to(&self, other: &DenseVec<f64>) -> DenseVec<f64> {
        let final_vec: &mut DenseVec<f64> = &mut other.clone();
        for column in self.data.axis_iter(Axis(1)) {
            let column = DenseVec::new(column.iter().map(|&x| x).collect());
            final_vec.sub_mut(&column.scale(other.inner(&column)));
        }
        //TODO: how can i avoid cloning if I want to mutate in place
        final_vec.clone()
    }
}
