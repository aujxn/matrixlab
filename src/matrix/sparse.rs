//
//    matrixlab, a library for working with sparse matricies
//    Copyright (C) 2019 Waylon Cude
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
//

use super::dense::DenseMatrix;
use super::MatrixElement;
use crate::error::Error;
use crate::iter::{ElementsIter, MatrixIter};
use crate::vector::{FloatVectorTrait, Vector, VectorTrait};
use rayon::prelude::*;
use std::fmt::{self, Display};
use std::ops::{Add, Mul};

#[derive(Clone, Debug, PartialEq, Copy)]
/// A point with i and j coordinates, as well as some data.
pub struct Element<A: MatrixElement>(pub usize, pub usize, pub A);

impl<A: MatrixElement> Element<A> {
    /// i is the row and j is the column index. These start at 0.
    pub fn new(i: usize, j: usize, data: A) -> Element<A> {
        Element(i, j, data)
    }
}

#[derive(Clone, Debug, PartialEq)]
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
/// indices will be equivalent.
///
/// - The column vector indices correspond to the data vector
/// indices. The value is the column of that element.
///
/// This storage format is efficient for memory usage and 
/// operations on sparse matrices. Storage of dense matrices
/// with this format should be avoided. Insertion into existing
/// sparse matrices is painfully inefficient, requiring shifting
/// of data and column values and adjustment of row values for
/// every member after the insertion.
pub struct Matrix<A: MatrixElement> {
    //The start of each row
    rows: Vec<usize>,
    //The data as one big array
    data: Vec<A>,
    //The column for each piece of data
    columns: Vec<usize>,
    num_rows: usize,
    num_columns: usize,
    //The '0' value of the data
    default: A,
}

impl<A: MatrixElement> Matrix<A> {
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
    /// let data = [(0, 0, 12), (3, 5, 4), (2, 2, 3) (1, 4, 42)];
    ///
    /// let elements: Vec<Element> = data.iter().map(|x| Element(x)).collect();
    /// let matrix = Matrix::new(4, 6, elements);
    ///
    /// assert_eq!(matrix.get(0, 0), Some(&12));
    /// assert_eq!(matrix.get(3, 5), Some(&4));
    /// assert_eq!(matrix.get(2, 2), Some(&3));
    /// assert_eq!(matrix.get(1, 4), Some(&42));
    /// ```
    ///
    /// Bad data:
    /// ```
    /// let out_of_bounds = vec![Element(3, 0, 10), Element(1, 1, 4)]
    /// let duplicates = vec![Element(1, 1, 1), Element(1, 1, 5)]
    ///
    /// let out_of_bounds = Matrix::new(3, 3, out_of_bounds);
    /// let duplicates = Matrix::new(3, 3, duplicates);
    ///
    /// assert_eq!(out_of_bounds, Err(Error::ElementOutOfBounds));
    /// assert_eq!(duplicates, Err(Error::DuplicateElements));
    /// ```
    pub fn new(
        num_rows: usize,
        num_columns: usize,
        mut points: Vec<Element<A>>,
    ) -> Result<Matrix<A>, Error> {
        //First we sort our points so we can insert them in order
        points.par_sort_unstable_by(|&Element(y1, x1, _), &Element(y2, x2, _)| {
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
            .map(|(&Element(y1, x1, _), &Element(y2, x2, _))| y1 == y2 && x1 == x2)
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
        let mut last_row_location = 0;
        let mut counter = 0;

        for Element(y, x, d) in points {
            if x > num_columns || y > num_rows {
                return Err(Error::ElementOutOfBounds);
            }
            data.push(d);
            columns.push(x);
            // If we've gotten to a new row
            if y != row_counter {
                // Fill in the rows with some extra copies
                let difference = y - row_counter;
                for _ in 1..difference {
                    rows.push(last_row_location);
                }
                // Push the location of our current row
                rows.push(counter);
                last_row_location = counter;
                row_counter = y;
            }
            counter += 1;
        }
        for _ in row_counter..=num_rows {
            rows.push(data.len());
        }

        Ok(Matrix {
            rows,
            data,
            columns,
            num_rows,
            num_columns,
            default: A::default(),
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
        mut points: Vec<Element<A>>,
    ) -> Result<Matrix<A>, Error> {
        //First we sort our points so we can insert them in order
        points.par_sort_unstable_by(|&Element(y1, x1, _), &Element(y2, x2, _)| {
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
        let mut last_row_location = 0;
        let mut counter = 0;

        for Element(i, j, v) in points {
            data.push(v);
            columns.push(j);
            // If we've gotten to a new row
            if i != row_counter {
                // Fill in the rows with some extra copies
                let difference = i - row_counter;
                for _ in 1..difference {
                    rows.push(last_row_location);
                }
                // Push the location of our current row
                rows.push(counter);
                last_row_location = counter;
                row_counter = i;
            }
            counter += 1;
        }
        for _ in row_counter..=num_rows {
            rows.push(data.len());
        }

        Ok(Matrix {
            rows,
            data,
            columns,
            num_rows,
            num_columns,
            default: A::default(),
        })
    }

    /// Returns the column (j) dimension
    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// Returns the row (i) dimension
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Returns a mutable reference to the element if it exists.
    /// Otherwise returns Err, unlike ['get'] which returns a 0
    /// for elements not in the matrix.
    ///
    /// ['get']: method.get
    ///
    /// # Examples
    ///
    /// ```
    /// let data = [(0, 0, 12), (3, 5, 4), (2, 2, 3) (1, 4, 42)];
    ///
    /// let elements: Vec<Element> = data.iter().map(|x| Element(x)).collect();
    /// let matrix = Matrix::new(4, 6, elements);
    ///
    /// if let Ok(val) = matrix.get_mut(0, 0) {
    ///     *val = 2
    ///     }
    ///
    /// if let Ok(val) = matrix.get_mut(3, 5) {
    ///     *val = 7
    ///     }
    ///
    /// let data: Vec<i8> = matrix.elements().map(|vals| vals).collect();
    ///
    /// assert_eq!(data, vec![2, 42, 3, 7]);
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
        }
        else {
            row_end = self.data.len();
        }

        //TODO: because data is stored sorted by column switch this to binary search
        for i in row_start..row_end {
            if *self.columns.get(i).unwrap() == column {
                return Ok(self.data.get_mut(i).unwrap());
            }
        }
        return Err(Error::NotFound)
    }

    /// Returns an iterator over all the nonzero elements of the array
    /// in order of row then column.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = [(0, 0, 12), (3, 5, 4), (2, 2, 3) (1, 4, 42)];
    /// let result = [(0, 0, 12), (1, 4, 42), (2, 2, 3), (3, 5, 4)];
    ///
    /// let elements: Vec<Element> = data.iter().map(|x| Element(x)).collect();
    /// let result: Vec<Element> = result.iter().map(|x| Element(x)).collect();
    /// let matrix = Matrix::new(4, 6, elements);
    ///
    /// let data: Vec<i8> = matrix.elements().map(|vals| vals).collect();
    ///
    /// assert_eq!(data, result);
    /// ```
    pub fn elements(&self) -> ElementsIter<A> {
        ElementsIter::new(&self)
    }

    /// Returns an iterator over all elements of the array, including the
    /// zero elements.
    /// # Examples
    ///
    /// ```
    /// let matrix = Matrix::new(3, 3, vec![Element(0, 0, 12), Element(2, 2, 4));
    ///
    /// let all_iter: Vec<i8> = matrix.all_elements()
    ///
    /// assert_eq!(all_iter.next(), Some(&Element(0, 0, 12)));
    /// assert_eq!(all_iter.next(), Some(&Element(0, 1, 0)));
    /// assert_eq!(all_iter.skip(5).next(), Some(&Element(2, 2, 4)));
    /// ```
    pub fn all_elements(&self) -> MatrixIter<A> {
        MatrixIter::new(&self)
    }

    /// Create a new matrix that is the transpose of this matrix
    pub fn transpose(&self) -> Matrix<A> {
        // Create a new matrix, inverted from our first matrix
        let points = self
            .elements()
            // Swap the x and y coordinates for each point
            .map(|Element(y, x, d)| Element(x, y, *d));
        Matrix::new(self.num_columns, self.num_rows, points.collect())
            .expect("Invalid matrix transpose")
    }

    /// Returns the number of nonzero entries in the matrix
    pub fn nnz(&self) -> usize {
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

    /// Returns a reference to the value. If no value exists then
    /// 0 (default) is returned.
    /// If out of bounds then error is returned.
    /// 
    /// # Examples
    ///
    /// ```
    /// let data = [(0, 0, 12), (3, 5, 4), (2, 2, 3) (1, 4, 42)];
    ///
    /// let elements: Vec<Element> = data.iter().map(|x| Element(x)).collect();
    /// let matrix = Matrix::new(4, 6, elements);
    ///
    /// assert_eq!(matrix.get(0, 0), Ok(&12));
    /// assert_eq!(matrix.get(3, 5), Ok(&4));
    /// assert_eq!(matrix.get(2, 2), Ok(&3));
    /// assert_eq!(matrix.get(1, 4), Ok(&42));
    /// assert_eq!(matrix.get(0, 1), Ok(&0));
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
        if row < max_row {
            row_end = *self.rows.get(row + 1).unwrap();
        }
        else {
            row_end = self.data.len();
        }

        //TODO: because data is stored sorted by column switch this to binary search
        for i in row_start..row_end {
            if *self.columns.get(i).unwrap() == column {
                return Ok(self.data.get(i).unwrap());
            }
        }
        return Ok(&self.default);
    }
}

impl <A: MatrixElement + std::ops::AddAssign> Matrix<A> {
    /// Calulates row sums and returns a vector with the sums
    pub fn row_sums(&self) -> std::vec::Vec<A> {
        let mut sums = vec![self.default; self.num_rows];
        self.elements().for_each(|Element(i, _, v)| sums[i] += *v);
        sums
    }
}

impl<A: MatrixElement + Mul<Output = A> + Add<Output = A>> Matrix<A> {
    /// Multiplication by another matrix, giving a result back
    pub fn safe_mul(&self, other: &Matrix<A>) -> Result<Matrix<A>, Error> {
        //Check to make sure the dimensions of our matrix match up
        if other.num_rows() != self.num_columns() {
            return Err(Error::SizeMismatch);
        }
        //Otherwise get on with the multiplication
        //TODO: reserve exactly enough space for this
        //let mut points = Vec::new();
        //We split other up by column, then do a bunch of multiplications
        //by vector
        let points = (0..other.num_columns)
            .into_par_iter()
            .map(|i| {
                //Get all points that are in the column
                let column = other
                    .elements()
                    .filter(|Element(_, x, _)| *x == i)
                    .map(|Element(i, j, val)| Element(i, j, *val))
                    .collect::<Vec<Element<A>>>();
                if let Ok(mut new_vec) = self.sparse_vec_mul(&column) {
                    //Modify all of the columns
                    new_vec
                        .iter_mut()
                        .for_each(|Element(_, column, _)| *column = i);
                    new_vec
                } else {
                    //Empty vector, as operation didn't finish
                    vec![]
                }
            })
            .flatten()
            .collect();

        Matrix::new(self.num_rows, other.num_columns, points)
    }

    fn sparse_vec_mul(&self, other: &Vector<Element<A>>) -> Result<Vector<Element<A>>, Error> {
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
                if let Some(Element(_, _, other_data)) =
                    other.iter().filter(|Element(y, _, _)| *y == *column).next()
                {
                    sum = sum + data * *other_data;
                }
            }
            //Check to see if we should push a value
            if sum != Default::default() {
                //Assume we're all in the same column
                let &Element(_, column, _) = other.get(0).ok_or(Error::MalformedInput)?;
                //Here we create the now element of our sparse vector.
                output.push(Element(row + 1, column, sum));
            }
        }
        Ok(output)
    }
    pub fn vec_mul(&self, other: &Vector<A>) -> Result<Vector<A>, Error> {
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
                sum = sum + data * other[self.columns[i] - 1];
            }
            output.push(sum);
        }
        Ok(output)
    }
}

//We can only do gmres with f64 types
impl Matrix<f64> {
    pub fn gmres(
        &self,
        b: Vector<f64>,
        max_iterations: usize,
        tolerance: f64,
        max_search_directions: usize,
    ) -> Result<Vector<f64>, Error> {
        // If the rows don't match up the error out straight away
        if self.num_columns() != self.num_rows() {
            return Err(Error::SizeMismatch);
        }
        //TODO: These maybe should be parameters to the function
        //^ It has been made so
        //let max_iterations = 1000;
        let mut i = 0;
        //Tolerance is 10^-6
        //let tolerance = 1.0/1000000.0;
        // Create our guess, the 0 vector
        let mut x: Vector<f64> = [0.0f64]
            .into_iter()
            .cycle()
            .take(self.num_columns())
            // Why do I need a cloned?
            .cloned()
            .collect();
        let mut r = b.sub(&(self * &x));
        let r_norm = r.norm();
        let final_norm = r_norm * tolerance;

        // Our initial search direction, the first column of P
        let p = r.normalize();
        let mut big_b = DenseMatrix::new(vec![self * &p]);
        let mut big_p = DenseMatrix::new(vec![p]);
        loop {
            let alpha = big_b.least_squares(&r)?;

            x = x.add(&big_p.vec_mul(&alpha)?);
            //println!("X: {:?}",x);

            r = r.sub(&big_b.vec_mul(&alpha)?);
            //println!("R: {:?}",r);
            let norm = r.norm();
            //println!("i: {} NORM: {}",i,norm);
            if norm < final_norm {
                return Ok(x);
            } else if i >= max_iterations {
                return Err(Error::ExceededIterations(x));
            }
            i += 1;

            if big_p.num_columns() < max_search_directions {
                let p = big_p.orthogonal_to(&r).normalize();
                big_b.add_column(self.vec_mul(&p)?);
                big_p.add_column(p);
            } else {
                //Restart
                let result = self * &x;
                r = b.sub(&result);
                //println!("RESTARTED: {:?}",&result);
                //let r_norm = r.norm();
                //let final_norm = r_norm * tolerance;

                // Our initial search direction, the first column of P
                let p = r.normalize();
                big_b = DenseMatrix::new(vec![self * &p]);
                big_p = DenseMatrix::new(vec![p]);
            }
        }
    }
}

// Multiplication by a scalar
//impl<A: Mul<Output=A> + Copy + Sized + Send + Sync> Mul<A> for Matrix<A> {
impl<A: Mul<Output = A> + MatrixElement> Mul<A> for Matrix<A> {
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
impl<A: Mul<Output = A> + Add<Output = A> + MatrixElement> Mul<&Vector<A>>
    for &Matrix<A>
{
    //Should this be an option or should it panic?
    type Output = Vector<A>;
    fn mul(self, other: &Vector<A>) -> Self::Output {
        //This just wraps around one of our safe functions
        self.vec_mul(other).unwrap()
    }
}

// Multiplication by a sparse vector
impl<A: Mul<Output = A> + Add<Output = A> + MatrixElement> Mul<&Vector<Element<A>>>
    for &Matrix<A>
{
    //Should this be an option or should it panic?
    type Output = Vector<Element<A>>;
    fn mul(self, other: &Vector<Element<A>>) -> Self::Output {
        self.sparse_vec_mul(other).unwrap()
    }
}

// Multiplication by another matrix
impl<A: Mul<Output = A> + Add<Output = A> + MatrixElement> Mul<&Matrix<A>>
    for &Matrix<A>
{
    //Should this be an option or should it panic?
    type Output = Matrix<A>;
    fn mul(self, other: &Matrix<A>) -> Self::Output {
        // This just caches out to our safe matrix multiplication
        // But we unwrap stuff to make it more ergonomic, and there's
        // safe calls if you don't want to crash
        self.safe_mul(other).unwrap()
    }
}

impl<A: MatrixElement + Display> Display for Matrix<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let elements: Vec<A> = self.all_elements().map(|Element(_, _, val)| *val).collect();
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
