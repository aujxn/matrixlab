/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use super::dense::DenseMatrix;
use crate::error::Error;
use crate::matrix::sparse_matrix_iter::{ElementsIter, MatrixIter, RowIter};
use crate::vector::dense::DenseVec;
use crate::vector::sparse::SparseVec;
use crate::{Element, MatrixElement};
use rayon::prelude::*;
use std::fmt::{self, Display};
use std::ops::{Add, Mul};

/// A matrix in CSR format. Here is a basic description of CSR:
///
/// - All the data is in one vector and is sorted first by row
/// and second by column.
///
/// - The index of the row vector corresponds to the row index
/// of the matrix and the value is the index of the data vector
/// that starts that row.
///
/// - If a row is empty then the values of two adjacent row
/// indices will be equivalent. Example: if index 13 and 14
/// both contain the value 75, row 12 ends at the 74th index
/// of the data array and row 14 begins at 75.
///
/// - The column vector indices correspond to the data vector
/// indices. The value is the column of that element.
///
/// This storage format is efficient for memory usage and arithmetic
/// operations on sparse matrices. Storage of dense matrices
/// with this format should be avoided. Insertion into existing
/// sparse matrices is painfully inefficient. Consider:
///
/// - Requires shifting of data and column values for every data
/// after the insertion to maintain order
///
/// - Increment of row values for every row after the inserted data
///
/// - Potential re-allocation and complete copy of all the data
///
/// Getting the value at a specific location is of complexity
/// log(n), where n is number of values in the row of interest.
#[derive(PartialEq, Debug, Clone)]
pub struct SparseMatrix<A: Element> {
    //The start of each row
    rows: Vec<usize>,
    //The data as one big array
    data: Vec<A>,
    //The column for each piece of data
    columns: Vec<usize>,
    num_rows: usize,
    num_columns: usize,
}

impl<A: Element> SparseMatrix<A> {
    /// Create a new matrix from a set of points and row/column dimensions.
    ///
    /// This constructor is checked. ['new_unsafe'] for a faster, more
    /// intrepid, option.
    ///
    /// ['new_unsafe']: method.new_unsafe
    ///
    /// # Examples
    ///
    /// A valid matrix:
    /// ```
    /// use matrixlab::error::Error;
    /// use matrixlab::MatrixElement;
    /// use matrixlab::matrix::sparse::SparseMatrix;
    ///
    /// let data = vec![(0usize, 0usize, 12i64), (3, 5, 4), (2, 2, 3), (1, 4, 42)];
    ///
    /// let elements: Vec<MatrixElement<i64>> = data
    ///     .iter()
    ///     .map(|(i, j, val)| MatrixElement(*i, *j, *val))
    ///     .collect();
    ///
    /// let matrix = SparseMatrix::new(4, 6, elements).unwrap();
    ///
    /// assert_eq!(matrix.get(0, 0), Ok(&12));
    /// assert_eq!(matrix.get(3, 5), Ok(&4));
    /// assert_eq!(matrix.get(2, 2), Ok(&3));
    /// assert_eq!(matrix.get(1, 4), Ok(&42));
    /// ```
    ///
    /// Bad data:
    /// ```
    /// use matrixlab::error::Error;
    /// use matrixlab::MatrixElement;
    /// use matrixlab::matrix::sparse::SparseMatrix;
    ///
    /// let out_of_bounds = vec![MatrixElement(3, 0, 10), MatrixElement(1, 1, 4)];
    /// let duplicates = vec![MatrixElement(1, 1, 1), MatrixElement(1, 1, 5)];
    ///
    /// let out_of_bounds = SparseMatrix::new(3, 3, out_of_bounds);
    /// let duplicates = SparseMatrix::new(3, 3, duplicates);
    ///
    /// assert_eq!(out_of_bounds, Err(Error::ElementOutOfBounds));
    /// assert_eq!(duplicates, Err(Error::DuplicateElements));
    /// ```
    pub fn new(
        num_rows: usize,
        num_columns: usize,
        mut points: Vec<MatrixElement<A>>,
    ) -> Result<SparseMatrix<A>, Error> {
        //First we sort our points so we can insert them in order
        points.par_sort_unstable_by(|&MatrixElement(y1, x1, _), &MatrixElement(y2, x2, _)| {
            if y1 != y2 {
                // Which row it's on is more important, so we check
                // for that first
                y1.cmp(&y2)
            } else {
                // But if they're on the same row we just compare
                // which column the points are in
                x1.cmp(&x2)
            }
        });

        //Now we check to make sure there aren't multiple entries for any point
        //NOTE: This is somewhat slow, but ensures we end up with a valid matrix
        if points
            .iter()
            .zip(points.iter().skip(1))
            .map(|(&MatrixElement(y1, x1, _), &MatrixElement(y2, x2, _))| y1 == y2 && x1 == x2)
            .any(|b| b)
        {
            return Err(Error::DuplicateElements);
        }

        // Allocate enough space for each of our arrays, we won't
        // have to deal with unexpected allocations this way
        let mut data: Vec<A> = Vec::with_capacity(points.len());
        let mut rows = Vec::with_capacity(num_rows + 1);
        // The first element is always a 0
        rows.push(0);
        let mut columns = Vec::with_capacity(points.len());

        // Then we insert those points into data, and fill in columns
        // and data
        //
        // We start at row 0
        let mut row_counter = 0;
        let mut counter = 0;

        for MatrixElement(i, j, v) in points {
            if j >= num_columns || i >= num_rows {
                return Err(Error::ElementOutOfBounds);
            }
            data.push(v);
            columns.push(j);
            // If we've gotten to a new row
            if i != row_counter {
                // Fill in the rows with some extra copies
                let difference = i - row_counter;
                for _ in 0..difference {
                    rows.push(counter);
                }
                row_counter = i;
            }
            counter += 1;
        }
        for _ in row_counter..num_rows {
            rows.push(data.len());
        }

        Ok(SparseMatrix {
            rows,
            data,
            columns,
            num_rows,
            num_columns,
        })
    }

    /// Create a new matrix from a set of points and some dimensions.
    ///
    /// # Warning
    ///
    /// This function does not verify that there are no duplicates in
    /// your matrix and everything will probably break if this happens,
    /// so make sure there are no duplicates before calling this.
    ///
    /// The checked version: ['new']
    /// ['new']: method.new
    ///
    /// # Panics!
    ///
    /// Panics if any of the elements are out of range.
    pub fn new_unsafe(
        num_rows: usize,
        num_columns: usize,
        mut points: Vec<MatrixElement<A>>,
    ) -> Self {
        //First we sort our points so we can insert them in order
        points.par_sort_unstable_by(|&MatrixElement(y1, x1, _), &MatrixElement(y2, x2, _)| {
            if y1 != y2 {
                // Which row it's on is more important, so we check
                // for that first
                y1.cmp(&y2)
            } else {
                // But if they're on the same row we just compare
                // which column the points are in
                x1.cmp(&x2)
            }
        });

        // Allocate enough space for each of our arrays, we won't
        // have to deal with unexpected allocations this way
        let mut data: Vec<A> = Vec::with_capacity(points.len());
        let mut rows = Vec::with_capacity(num_rows + 1);
        // The first element is always a 0
        rows.push(0);
        let mut columns = Vec::with_capacity(points.len());

        // Then we insert those points into data, and fill in columns
        // and data
        //
        // We start at row 0
        let mut row_counter = 0;
        let mut counter = 0;

        for MatrixElement(i, j, v) in points {
            data.push(v);
            columns.push(j);
            // If we've gotten to a new row
            if i != row_counter {
                // Fill in the rows with some extra copies
                let difference = i - row_counter;
                for _ in 0..difference {
                    rows.push(counter);
                }
                row_counter = i;
            }
            counter += 1;
        }
        for _ in row_counter..num_rows {
            rows.push(data.len());
        }

        SparseMatrix {
            rows,
            data,
            columns,
            num_rows,
            num_columns,
        }
    }

    /// Creates a sparse matrix from constructed data.
    /// The length of rows should equal the number of rows
    /// and the length of columns and data should be equal.
    ///
    /// This function is unchecked so if malformed data is
    /// provided behaviour is undefined.
    ///
    /// This method should be used to avoid the overhead of
    /// sorting data that the other constructors have.
    ///
    /// Methods in this library expect data in each row to
    /// be sorted by column.
    pub fn new_csr(
        rows: Vec<usize>,
        columns: Vec<usize>,
        num_columns: usize,
        data: Vec<A>,
    ) -> Self {
        let num_rows = rows.len();
        SparseMatrix {
            rows,
            data,
            columns,
            num_rows,
            num_columns,
        }
    }

    /// Returns the column (j) dimension
    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// Returns the row (i) dimension
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Determines if the matrix is square
    pub fn is_square(&self) -> bool {
        self.num_rows == self.num_columns
    }

    /// Returns a mutable reference to the element if it exists.
    ///
    /// # Errors
    /// NotFound and ElementOutOfBounds
    ///
    /// # Examples
    ///
    /// ```
    /// use matrixlab::error::Error;
    /// use matrixlab::MatrixElement;
    /// use matrixlab::matrix::sparse::SparseMatrix;
    ///
    /// let data = [(0usize, 0usize, 12i64), (3, 5, 4), (2, 2, 3), (1, 4, 42)];
    ///
    /// let elements: Vec<MatrixElement<i64>> = data
    ///     .iter()
    ///     .map(|(i, j, val)| MatrixElement(*i, *j, *val))
    ///     .collect();
    /// let mut matrix = SparseMatrix::new(4, 6, elements).unwrap();
    ///
    /// if let Ok(val) = matrix.get_mut(0, 0) {
    ///     *val = 2
    ///     }
    ///
    /// if let Ok(val) = matrix.get_mut(3, 5) {
    ///     *val = 7
    ///     }
    ///
    /// let data: Vec<i64> = matrix
    ///     .elements()
    ///     .map(|(_, _, val)| *val)
    ///     .collect();
    ///
    /// assert_eq!(data, vec![2i64, 42, 3, 7]);
    /// assert_eq!(matrix.get_mut(1, 1), Err(Error::NotFound));
    /// assert_eq!(matrix.get_mut(4, 4), Err(Error::ElementOutOfBounds));
    /// assert_eq!(matrix.get_mut(2, 6), Err(Error::ElementOutOfBounds));
    /// ```
    pub fn get_mut(&mut self, row: usize, column: usize) -> Result<&mut A, Error> {
        //check if request is valid
        let max_row = self.num_rows - 1;
        if row > max_row || column >= self.num_columns {
            return Err(Error::ElementOutOfBounds);
        }

        //index range to look for the data in
        //unwrap can't panic here because manual bounds check
        let row_start = *self.rows.get(row).unwrap();
        let row_end;
        if row < max_row {
            row_end = *self.rows.get(row + 1).unwrap();
        } else {
            row_end = self.data.len();
        }

        //TODO: because data is stored sorted by column switch this to binary search
        for i in row_start..row_end {
            if *self.columns.get(i).unwrap() == column {
                return Ok(self.data.get_mut(i).unwrap());
            }
        }
        return Err(Error::NotFound);
    }

    /// Returns an iterator over all the nonzero elements of the array
    /// in order of row then column.
    ///
    /// # Examples
    ///
    /// ```
    /// use matrixlab::error::Error;
    /// use matrixlab::MatrixElement;
    /// use matrixlab::matrix::sparse::SparseMatrix;
    ///
    /// let data = vec![(0usize, 0usize, 12i8), (3, 5, 4), (2, 2, 3), (1, 4, 42)];
    /// let result = vec![(0usize, 0usize, 12i8), (1, 4, 42), (2, 2, 3), (3, 5, 4)];
    ///
    /// let elements: Vec<MatrixElement<i8>> = data
    ///     .iter()
    ///     .map(|(i, j, val)| MatrixElement(*i, *j, *val))
    ///     .collect();
    /// let matrix = SparseMatrix::new(4, 6, elements).unwrap();
    ///
    /// let data: Vec<(usize, usize, i8)> = matrix
    ///     .elements()
    ///     .map(|(i, j, val)| (i, j, *val))
    ///     .collect();
    ///
    /// assert_eq!(data, result);
    /// ```
    pub fn elements(&self) -> ElementsIter<A> {
        ElementsIter::new(&self)
    }

    /// Returns an iterator over the rows of the matrix as slices.
    pub fn row_iter(&self) -> RowIter<A> {
        RowIter::new(&self)
    }

    /// Returns an iterator over all elements of the matrix, including the
    /// zero elements. The elements are wrapped in an option.
    ///
    /// # Examples
    ///
    /// ```
    /// use matrixlab::error::Error;
    /// use matrixlab::MatrixElement;
    /// use matrixlab::matrix::sparse::SparseMatrix;
    ///
    /// let matrix = SparseMatrix::new(3, 3,
    ///     vec![MatrixElement(0usize, 0usize, 12i64), MatrixElement(2, 2, 4)])
    ///     .unwrap();
    ///
    /// let mut all_iter = matrix.all_elements();
    ///
    /// assert_eq!(all_iter.next(), Some((0, 0, Some(&12))));
    /// assert_eq!(all_iter.next(), Some((0, 1, None)));
    /// assert_eq!(all_iter.skip(6).next(), Some((2, 2, Some(&4))));
    /// ```
    pub fn all_elements(&self) -> MatrixIter<A> {
        MatrixIter::new(&self)
    }

    /// Create a new matrix that is the transpose of this matrix
    // TODO: for some multiplicaions is is better to transpose into CSC.
    // transposing into CSC is also free, so that option should be available.
    pub fn transpose(&self) -> SparseMatrix<A> {
        // Create a new matrix, inverted from our first matrix
        let points = self
            .elements()
            // Swap the x and y coordinates for each point
            .map(|(y, x, d)| MatrixElement(x, y, *d));
        SparseMatrix::new(self.num_columns, self.num_rows, points.collect())
            .expect("Invalid matrix transpose")
    }

    /// Returns the number of nonzero entries in the matrix
    pub fn num_nonzero(&self) -> usize {
        self.data.len()
    }

    /// This method returns the underlying data array of the matrix
    pub fn get_data(&self) -> &[A] {
        &self.data
    }

    /// This method returns the underlying column array of the matrix
    pub fn get_columns(&self) -> &[usize] {
        self.columns.as_slice()
    }

    /// This method returns the underlying row array of the matrix
    pub fn get_rows(&self) -> &[usize] {
        self.rows.as_slice()
    }

    /// Returns a reference to the value, if it exists.
    ///
    /// # Errors
    /// NotFound, ElementOutOfBounds
    ///
    /// # Examples
    ///
    /// ```
    /// use matrixlab::error::Error;
    /// use matrixlab::MatrixElement;
    /// use matrixlab::matrix::sparse::SparseMatrix;
    ///
    /// let data = vec![
    ///     (0usize, 0usize, 12i64),
    ///     (3, 5, 4), (2, 2, 3),
    ///     (1, 4, 42)];
    ///
    /// let elements: Vec<MatrixElement<i64>> = data
    ///     .iter()
    ///     .map(|(i, j, val)| MatrixElement(*i, *j, *val))
    ///     .collect();
    ///
    /// let matrix = SparseMatrix::new(4, 6, elements).unwrap();
    ///
    /// assert_eq!(matrix.get(0, 0), Ok(&12));
    /// assert_eq!(matrix.get(3, 5), Ok(&4));
    /// assert_eq!(matrix.get(2, 2), Ok(&3));
    /// assert_eq!(matrix.get(1, 4), Ok(&42));
    /// assert_eq!(matrix.get(0, 1), Err(Error::NotFound));
    /// assert_eq!(matrix.get(4, 4), Err(Error::ElementOutOfBounds));
    /// assert_eq!(matrix.get(2, 6), Err(Error::ElementOutOfBounds));
    /// ```
    pub fn get(&self, row: usize, column: usize) -> Result<&A, Error> {
        //check if request is valid
        let max_row = self.num_rows - 1;
        if row > max_row || column >= self.num_columns {
            return Err(Error::ElementOutOfBounds);
        }

        //index range to look for the data in
        //unwrap can't panic here because manual bounds check
        let row_start = *self.rows.get(row).unwrap();
        let row_end;
        // TODO: I think I did this and it isn't necesarry because the
        // rows vector has an extra value to account for this case. Not
        // completely sure.
        if row < max_row {
            row_end = *self.rows.get(row + 1).unwrap();
        } else {
            row_end = self.data.len();
        }

        //TODO: because data is stored sorted by column switch this to binary search
        for i in row_start..row_end {
            if *self.columns.get(i).unwrap() == column {
                return Ok(self.data.get(i).unwrap());
            }
        }
        return Err(Error::NotFound);
    }
}

impl<A: Element + std::ops::AddAssign + Default> SparseMatrix<A> {
    /// Calulates row sums and returns a vector with the sums
    pub fn row_sums(&self) -> std::vec::Vec<A> {
        let mut sums = vec![A::default(); self.num_rows];
        self.elements().for_each(|(i, _, v)| sums[i] += *v);
        sums
    }
}

impl<A: Element> SparseMatrix<A> {
    /// Gets the data of a row of the matrix
    /// # Panics:
    /// When an out of range row is provided
    pub fn get_row(&self, row: usize) -> (&[A], &[usize]) {
        (&self.data[self.rows[row]..self.rows[row + 1]], &self.columns[self.rows[row]..self.rows[row + 1]])
    }
}

impl<A: Element + Mul<Output = A> + Add<Output = A> + Default> SparseMatrix<A> {
    /* TODO:
    pub fn sparse_mat_mul(&self, other: &SparseMatrix<A>) -> SparseMatrix<A> {
    }
    */

    /// Multiplication of  another matrix, giving a result back
    pub fn safe_sparse_mat_mul(&self, other: &SparseMatrix<A>) -> Result<SparseMatrix<A>, Error> {
        //Check to make sure the dimensions of our matrix match up
        if other.num_rows() != self.num_columns() {
            return Err(Error::SizeMismatch);
        }

        // number of rows in the results will be same as rows of self
        let num_rows = self.num_rows();
        let num_columns = other.num_columns();
        let mut rows = Vec::with_capacity(self.rows.len());
        rows.push(0);
        //how much capacity for resulting data?
        let mut data = Vec::with_capacity(self.data.len() + other.data.len());
        let mut columns = Vec::with_capacity(self.data.len() + other.data.len());

        //TODO: change to par iterators for maybe speedup?
        for i in 0..num_rows {
            let left = self.get_row(i);
            for j in 0..num_columns {
                let mut result: Option<A> = None;
                for (left_data, left_col) in left.0.iter().zip(left.1.iter()) {
                    if let Ok(right_data) = other.get(*left_col, j) {
                        let product = *left_data * *right_data;
                        match result {
                            None => result = Some(product),
                            Some(sum) => result = Some(sum + product),
                        }
                    }
                }
                if let Some(val) = result {
                    data.push(val);
                    columns.push(j);
                }
            }
            rows.push(data.len());
        }

        Ok(SparseMatrix {
            num_rows,
            num_columns,
            rows,
            data,
            columns,
        })

        //Otherwise get on with the multiplication
        //TODO: reserve exactly enough space for this
        //let mut points = Vec::new();
        //
        //right now this is really bad and does tons of allocs
        //compiler is unlikely to optimize the temp SparseVec's
        //from allocating a ton of vectors
        /*
        let points = (0..other.num_columns)
            .into_par_iter()
            .map(|j| {
                //Get all points that are in the column
                let column = other
                    .elements()
                    .filter(|(_, x, _)| *x == j)
                    .map(|(i, _, val)| (i, *val))
                    .collect();
                let column = SparseVec::new_unsafe(column, other.num_columns);
                if let Ok(new_col) = self.sparse_vec_mul(&column) {
                    //Modify all of the columns
                    new_col
                        .get_data()
                        .iter()
                        .map(|&(i, val)| MatrixElement(i, j, val))
                        .collect()
                } else {
                    //Empty vector, as operation didn't finish
                    vec![]
                }
            })
            .flatten()
            .collect();

        SparseMatrix::new(self.num_rows, other.num_columns, points)
        */
    }

    /* TODO
    pub fn dense_mat_mul(&self, other: &SparseMatrix<A>) -> DenseMatrix<A> {
    }

    pub fn safe_dense_mat_mul(&self, other: &SparseMatrix<A>) -> Result<DenseMatrix<A>, Error> {
    }
    */

    /// Multiplication of sparse matrix and sparse vector
    fn sparse_vec_mul(&self, other: &SparseVec<A>) -> Result<SparseVec<A>, Error> {
        //Probably TODO: check to see if any elements of our vector
        //are out of bounds, the current bounds check is super limited
        //
        //Check to make sure the the length matches up with row size
        if other.len() > self.num_columns() {
            return Err(Error::SizeMismatch);
        }
        //Otherwise get on with the multiplication
        let mut output = Vec::new();

        let row_ranges = self.rows.iter().zip(self.rows.iter().skip(1));
        for (row, (&start, &end)) in row_ranges.enumerate() {
            let mut sum: A = Default::default();
            //for (column,&data) in  (1..).zip(self.data[start..end].iter()) {
            for (column, &data) in self.columns[start..end]
                .iter()
                .zip(self.data[start..end].iter())
            {
                //We have to traverse to find the right element
                if let Some((_, other_data)) = other
                    .get_data()
                    .iter()
                    .filter(|&(i, _)| *i == *column)
                    .next()
                {
                    sum = sum + data * *other_data;
                }
            }
            //Check to see if we should push a value
            if sum != Default::default() {
                //Here we create the now element of our sparse vector.
                output.push((row, sum));
            }
        }
        Ok(SparseVec::new_unsafe(output, self.num_columns()))
    }

    /* TODO:
    fn safe_sparse_vec_mul(&self, other: &SparseVec<A>) -> Result<SparseVec<A>, Error> {
    }
    */

    /// Multiplication of sparse matrix and dense vector
    pub fn dense_vec_mul(&self, other: &DenseVec<A>) -> Result<DenseVec<A>, Error> {
        //Check to make sure the the length matches up with row size
        if other.len() != self.num_columns() {
            return Err(Error::SizeMismatch);
        }
        //Otherwise get on with the multiplication
        let mut output: Vec<A> = Vec::with_capacity(self.num_rows());
        //iter::repeat( Default::default()).take(self.num_rows).collect();

        let row_ranges = self.rows.iter().zip(self.rows.iter().skip(1));
        for (&start, &end) in row_ranges {
            let mut sum: A = Default::default();
            for (i, &data) in (start..end).zip(self.data[start..end].iter()) {
                sum = sum + data * other.get_data()[self.columns[i]];
            }
            output.push(sum);
        }
        Ok(DenseVec::new(output))
    }

    /* TODO:
    pub fn safe_dense_vec_mul(&self, other: &DenseVec<A>) -> Result<DenseVec<A>, Error> {
    }
    */
}

/* gmres is getting rebuilt to be fast fast fast
//We can only do gmres with f64 types
impl SparseMatrix<f64> {
    /// Solves a linear system using the generalized minimal residual method.
    /// Only implemented for SparseMatrix<f64>
    ///
    /// # Example
    ///
    /// ```
    /// use matrixlab::MatrixElement;
    /// use matrixlab::matrix::sparse::SparseMatrix;
    /// use matrixlab::vector::dense::DenseVec;
    ///
    /// let elements = vec![MatrixElement(0, 0, 2f64), MatrixElement(1, 1, 2f64), MatrixElement(0, 1, 1.0)];
    /// let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();
    ///
    /// let result = mat
    ///     .gmres(DenseVec::new(vec![3.0, 2.0]), 1000, 1.0 / 1000000.0, 50)
    ///     .unwrap();
    ///     assert_eq!(result, DenseVec::new(vec![1.0, 1.0]));
    /// ```
    pub fn gmres(
        &self,
        b: DenseVec<f64>,
        max_iterations: usize,
        tolerance: f64,
        max_search_directions: usize,
    ) -> Result<DenseVec<f64>, Error> {
        // If the rows don't match up error out straight away
        if self.num_columns() != self.num_rows() {
            return Err(Error::SizeMismatch);
        }
        let mut i = 0;
        let m = self.num_columns();
        // Create our guess, the 0 vector
        let mut x: DenseVec<f64> = DenseVec::new(vec![0.0; m]);
        let mut residual = b.clone();       // b is first residual because Ax = 0
        let residual_norm = residual.norm();
        let final_norm = residual_norm * tolerance;

        // Our initial search direction, the first column of P
        // This is just the normalized first residual vector
        let initial_p = residual.normalize();
        println!("{:?}", initial_p);

        // These are dense matrices but represented as Vectors of columns
        // This is easier than using the DenseMatrix abstraction because
        // each iteration a column is added to each matrix.
        let mut big_b: Vec<DenseVec<f64>> = vec![self * &initial_p];
        let mut big_p: Vec<DenseVec<f64>> = vec![initial_p];

        loop {
            let cols = big_b.len();
            let B = DenseMatrix::from_columns(m, cols, big_b.clone())?;
                
            let p = DenseMatrix::from_columns(m, big_p.len(), big_p.clone())?;
            let alpha = B.least_squares(&residual)?;
            if i == 0 {
                println!("least squares: {:?}", alpha);
            }

            x = x.add(&p.dense_vec_mul(&alpha));

            residual = residual.sub(&B.dense_vec_mul(&alpha));
            //println!("R: {:?}",r);
            let norm = residual.norm();
            //println!("i: {} NORM: {}",i,norm);
            if norm < final_norm {
                return Ok(x);
            } else if i >= max_iterations {
                return Err(Error::ExceededIterations(x.get_data().clone()));
            }
            i += 1;

            if cols < max_search_directions {
                let p = p.orthogonal_to(&residual).normalize();
                big_b.push(self * &p);
                big_p.push(p);
            } else {
                //Restart
                let result = self * &x;
                residual = b.sub(&result);
                //println!("RESTARTED: {:?}",&result);
                //let r_norm = r.norm();
                //let final_norm = r_norm * tolerance;

                // Our initial search direction, the first column of P
                let p = residual.normalize();
                big_b = vec![self * &p];
                big_p = vec![p];
            }
        }
    }
}
*/

// Multiplication by a scalar
//impl<A: Mul<Output=A> + Copy + Sized + Send + Sync> Mul<A> for SparseMatrix<A> {
impl<A: Mul<Output = A> + Element + Default> Mul<A> for SparseMatrix<A> {
    type Output = Self;
    fn mul(mut self, other: A) -> Self {
        //Multiply all of the data by other
        //Um this does a ton of copies
        //But it's probably okay if we're working with numbers
        //TODO: benchmark against nonparallel version
        self.data
            .par_iter_mut()
            .for_each_with(other, |o, x| *x = o.clone() * x.clone());
        self
    }
}

// Multiplication by a vector
impl<A: Mul<Output = A> + Add<Output = A> + Element + Default> Mul<&DenseVec<A>>
    for &SparseMatrix<A>
{
    //Should this be an option or should it panic?
    type Output = DenseVec<A>;
    fn mul(self, other: &DenseVec<A>) -> Self::Output {
        //This just wraps around one of our safe functions
        self.dense_vec_mul(other).unwrap()
    }
}

// Multiplication by a sparse vector
impl<A: Mul<Output = A> + Add<Output = A> + Element + Default> Mul<&SparseVec<A>>
    for &SparseMatrix<A>
{
    //Should this be an option or should it panic?
    type Output = SparseVec<A>;
    fn mul(self, other: &SparseVec<A>) -> Self::Output {
        self.sparse_vec_mul(other).unwrap()
    }
}

// Multiplication by another matrix
impl<A: Mul<Output = A> + Add<Output = A> + Element + Default> Mul<&SparseMatrix<A>>
    for &SparseMatrix<A>
{
    //Should this be an option or should it panic?
    type Output = SparseMatrix<A>;
    fn mul(self, other: &SparseMatrix<A>) -> Self::Output {
        // This just caches out to our safe matrix multiplication
        // But we unwrap stuff to make it more ergonomic, and there's
        // safe calls if you don't want to crash
        self.safe_sparse_mat_mul(other).unwrap()
    }
}

impl<A: Element + Display + Default> Display for SparseMatrix<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let elements: Vec<A> = self.all_elements().map(|(_, _, val)| {
            match val {
                Some(val) => *val,
                None => A::default(),
            }
        }).collect();
        //We want to print a row out at a time
        let chunks = elements.chunks(self.num_columns());
        for chunk in chunks {
            write!(f, "[")?;
            for (i, element) in chunk.iter().enumerate() {
                write!(f, "{}", element)?;
                //Print all except for the trailing comma
                if i != chunk.len() - 1 {
                    write!(f, ",")?;
                }
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}
