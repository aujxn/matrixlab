use crate::error::Error;
use crate::Element;
use crate::MatrixElement;
use crate::vector::dense::DenseVec;
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};
use ndarray::Array2;

#[derive(PartialEq, Debug)]
/// A dense matrix is a vector of dense vectors.
/// Each dense vector is a column of the matrix.
pub struct DenseMatrix<A> {
    data: Array2<A>,
}

impl<A> DenseMatrix<A> {
    /// Creates a matrix from a vector of columns.
    pub fn from_columns(num_rows: usize, num_columns: usize, columns: Vec<Vec<A>>) -> Result<DenseMatrix<A>, Error> {
    }

    /// Creates a matrix from a vector of rows.
    pub fn from_rows(num_rows: usize, num_columns: usize, rows: Vec<Vec<A>>) -> Result<DenseMatrix<A>, Error> {
    }

    /// Creates a dense matrix from a set of elements. If multiple non-zero
    /// elements are found at the same locations the sum is taken.
    pub fn from_triplets(num_rows: usize, num_columns: usize, elements: Vec<MatrixElement<A>>) -> Result<DenseMatrix<A>, Error> {
    }

    /// Creates a matrix from a 2 dimensional array created with the
    /// ndarray crate.
    pub fn from_array2(data: Array2<A>) -> DenseMatrix<A> {
        DenseMatrix { data }
    }

    /// Replaces an entire column of the matrix
    pub fn replace_column(&mut self, column: Vec<A>, index: usize) {
    }

    /// Replaces an entire row of the matrix
    pub fn replace_row(&mut self, row: Vec<A>, index: usize) {
    }

    pub fn get_mut_data(&mut self) -> &mut Array2<A> {
        &mut self.data
    }

    pub fn get_data(&mut self) -> Array2<A> {
        &self.data
    }
}

//Maybe todo: should this be copy?
impl<A: Clone> DenseMatrix<A> {
    pub fn transpose(&self) -> DenseMatrix<A> {
        // Set up the columns for our new matrix
        let mut columns = Vec::with_capacity(self.num_rows());
        for _ in 0..self.num_rows() {
            columns.push(Vec::with_capacity(self.num_columns()));
        }

        // Set up the elements of the columns of our new array
        for column in self.columns.iter() {
            for (i, entry) in column.iter().enumerate() {
                columns[i].push(entry.clone());
            }
        }

        DenseMatrix::new(columns)
    }

    /// Creates a matrix from a vector of rows
    pub fn from_rows(rows: Vec<DenseVec<A>>) -> DenseMatrix<A> {
        DenseMatrix { columns: rows }.transpose()
    }
}

//TODO: Make this generic?
impl<A: Element + Mul<Output = A> + Add<Output = A> + Sub<Output = A>> DenseMatrix<A> {
    pub fn scale(&self, other: &A) -> DenseMatrix<A> {
        let columns = self
            .columns
            .par_iter()
            .map_with(other, |&mut o, column| {
                column.iter().map(|e| *o * *e).collect()
            })
            .collect();
        DenseMatrix::new(columns)
    }
    
    pub fn dense_vec_mul(&self, other: &DenseVec<A>) -> DenseVec<A> {
        self
            .columns
            .par_iter()
            .zip(other.par_iter())
            // Is there any way to make this a normal iterator
            // and still be able to flatten?
            // Is collecting it slow?
            .map(|(column, scale)| column.iter().map(|x| *x * *scale).collect::<Vec<A>>())
            .reduce(
                || {
                    [Default::default()]
                        .into_iter()
                        .cycle()
                        .take(self.num_rows())
                        .cloned()
                        .collect()
                },
                |x, y| x.add(&y),
            )
        //.collect()
    }

    pub fn safe_dense_vec_mul(&self, other: &DenseVec<A>) -> DenseVec<A> {
    }

    pub fn sparse_vec_mul(&self, other: &SparseVec<A>) -> DenseVec<A> {
    }

    pub fn safe_sparse_vec_mul(&self, other: &SparseVec<A>) -> DenseVec<A> {
    }

    pub fn safe_dense_mat_mul(&self, other: &DenseMatrix<A>) -> Result<DenseMatrix<A>, Error> {
        if self.num_columns() != other.num_rows() {
            return Err(Error::SizeMismatch);
        }

        let new_cols = other
            .columns
            .par_iter()
            .map_with(self, |&mut s, col| s.vec_mul(col))
            .collect::<Result<Vec<Vec<A>>, Error>>()?;

        Ok(DenseMatrix::new(new_cols))
    }
}

impl DenseMatrix<f64> {
    /// This takes an upper triangular matrix, and solves it to
    /// equal b
    pub fn backsolve(&self, b: &DenseVec<f64>) -> DenseVec<f64> {
        // Start off with a copy of b, to modify to create our solutions
        let mut solutions: Vec<f64> = b.clone();
        // Start with the last column
        for (i, column) in self.columns.iter().rev().enumerate() {
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
        solutions
    }

    /// This solves for B*y = r
    pub fn least_squares(&self, r: &DenseVec<f64>) -> Result<DenseVec<f64>, Error> {
        //Solve for Q, for our QR factorization
        let q = self.factor_q();
        let q_transpose = q.transpose();
        let rhs = q_transpose.vec_mul(&r)?;
        let r = q_transpose
            .safe_mul(self)
            .expect("Error in least squares, multiplication failed");;

        //Now solve for Ra = rhs, then return a
        Ok(r.backsolve(&rhs))
    }

    pub fn factor_q(&self) -> DenseMatrix<f64> {
        let mut q_vectors: Vec<DenseVec<f64>> = Vec::with_capacity(self.num_columns());
        for column in self.columns.iter() {
            let mut maybe_q = column.clone();
            for orthogonal in q_vectors.iter() {
                let c = column.inner(orthogonal);
                // subtract c_n * q_n
                maybe_q = maybe_q.sub(&orthogonal.scale(c));
            }
            let q = maybe_q.normalize();
            q_vectors.push(q);
        }

        DenseMatrix::new(q_vectors)
    }

    /// Returns a new vector, orthogonal to all vectors currently in the
    /// array and to the other vector
    pub fn orthogonal_to(&self, other: &DenseVec<f64>) -> DenseVec<f64> {
        let mut final_vec = other.clone();
        for column in self.columns.iter() {
            final_vec = final_vec.sub(&column.scale(other.inner(column)));
        }
        final_vec
    }
}
