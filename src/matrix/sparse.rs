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

use std::ops::{Mul, Add};
use std::fmt::{self,Display};
use rayon::prelude::*;
use crate::iter::{ElementsIter,MatrixIter};
use crate::error::Error;
use crate::vector::{VectorTrait,Vector,FloatVectorTrait};
use super::dense::DenseMatrix;
use super::MatrixElement;


#[derive(Clone,Debug,PartialEq,Copy)]
/// A point with y and x coordinates, as well as some data
pub struct Element<A: MatrixElement>(pub usize,pub usize,pub A);

#[derive(Clone,Debug,PartialEq)]
///A matrix in CSR format
pub struct Matrix<A: MatrixElement> {
    //The start of each row
    rows: Vec<usize>,
    //The data as one big array
    data: Vec<A>,
    //The column for each piece of data
    columns: Vec<usize>,
    num_rows: usize,
    num_columns: usize
}
impl<A: MatrixElement> Matrix<A> {
    /// Create a new matrix from a set of points and some dimensions
    pub fn new(
        num_rows: usize, 
        num_columns: usize,
        mut points: Vec<Element<A>>) 
    -> Result<Matrix<A>,Error> {
        //First we sort our points so we can insert them in order
        points.par_sort_unstable_by(
            |&Element(y1,x1,_),&Element(y2,x2,_)| {
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
        if points.iter().zip(points.iter().skip(1))
            .map(|(&Element(y1,x1,_),&Element(y2,x2,_))| y1 == y2 && x1 == x2)
            .any(|b| b)
        {
            return Err(Error::DuplicateElements)
        }

        // Allocate enough space for each of our arrays, we won't
        // have to deal with unexpected allocations this way
        let mut data: Vec<A> = Vec::with_capacity(points.len());
        let mut rows = Vec::with_capacity(num_rows+1);
        // The first element is always a 0
        rows.push(0);
        let mut columns = Vec::with_capacity(points.len());

        // Then we insert those points into data, and fill in columns
        // and data
        //
        // We start at row 1
        let mut row_counter = 1;
        let mut last_row_location = 0;
        let mut counter = 0;
        for Element(y,x,d) in points {
            if x > num_columns || y > num_rows {
                return Err(Error::ElementOutOfBounds);
            }
            data.push(d);                                    
            columns.push(x);
            // If we've gotten to a new row
            if y != row_counter {
                // Fill in the rows with some extra copies
                let difference = y - row_counter;
                for _ in 1 .. difference {
                    rows.push(last_row_location);
                }
                // Push the location of our current row
                rows.push(counter);
                last_row_location = counter;
                row_counter = y;
            }

            counter += 1;
        }
        for _ in row_counter ..= num_rows {
            rows.push(data.len());
        }

        //rows.push(data.len());

        Ok(Matrix {
            rows,
            data,
            columns,
            num_rows,
            num_columns
        })
    }

    pub fn num_columns(&self) -> usize {
        self.num_columns
    }
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }
    //TODO: rework to return a 0
    /// Returns a reference to
    /// the element if it exists, otherwise returns a None
    /// which is usually equivalent to a 0
    pub fn get(&self,row: usize, column: usize) -> Option<&A> {
        let row_start = *self.rows.get(row-1)?;
        //This should always return some
        //Otherwise error out
        //Maybe it's better to unwrap here?
        let row_end   = *self.rows.get(row)?;

        for i in row_start .. row_end {
            //Maybe todo: unwrap again?
            if *self.columns.get(i)? == column {
                return self.data.get(i);
            }
        }
        return None;
    }
    /// Returns a mutable reference to
    /// the element if it exists, otherwise returns a None
    /// which is usually equivalent to a 0
    pub fn get_mut(&mut self,row: usize, column: usize) -> Option<&mut A> {
        let row_start = *self.rows.get(row)?;
        //This should always return some
        //Otherwise error out
        //Maybe it's better to unwrap here?
        let row_end   = *self.rows.get(row+1)?;

        for i in row_start .. row_end {
            //Maybe todo: unwrap again?
            if *self.columns.get(i)? == column {
                return self.data.get_mut(i);
            }
        }
        return None;
    }

    /// Returns an iterator over all the nonzero elements of the array
    pub fn elements(&self) -> ElementsIter<A> {
        ElementsIter::new(&self)
    }
    /// Returns an iterator over all elements of the array, including the
    /// zero elements
    pub fn all_elements(&self) -> MatrixIter<A> {
        MatrixIter::new(&self)
    }
    /// Create a new matrix that is the transpose of this matrix
    pub fn transpose(&self) -> Matrix<A> {
        // Create a new matrix, inverted from our first matrix
        let points = self.elements()
                         // Swap the x and y coordinates for each point
                         .map(|Element(y,x,d)| Element(x,y,d));
        Matrix::new(self.num_columns,self.num_rows,points.collect())
            .expect("Invalid matrix transpose")
    }
    /// Returns the number of nonzero entries in the matrix
    pub fn number_of_nonzero(&self) -> usize {
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

}
impl<A: MatrixElement + Mul<Output=A> + Add<Output=A> + Default> Matrix<A> {
    /// Multiplication by another matrix, giving a result back
    pub fn safe_mul(&self, other: &Matrix<A>) -> Result<Matrix<A>,Error> {
        //Check to make sure the dimensions of our matrix match up
        if  other.num_rows() != self.num_columns() {
            return Err(Error::SizeMismatch);
        }
        //Otherwise get on with the multiplication
        //TODO: reserve exactly enough space for this
        //let mut points = Vec::new();
        //We split other up by column, then do a bunch of multiplications
        //by vector
        let points = (1 .. other.num_columns + 1).into_par_iter().map(|i| {
            //Get all points that are in the column
            let column = other.elements()
                              .filter(|Element(_,x,_)| *x == i)
                              .collect::<Vec<Element<A>>>();
            if let Ok(mut new_vec) = self.sparse_vec_mul(&column) {
                //Modify all of the columns
                new_vec.iter_mut().for_each(
                    |Element(_,column,_)| *column = i);
                new_vec
            } else {
                //Empty vector, as operation didn't finish
                vec![]
            }

        //Um I dont think I should have to flatten twice
        }).flatten().collect();

        Matrix::new(self.num_rows,other.num_columns,points)
        
    }

    fn sparse_vec_mul(&self, other: &Vector<Element<A>>) -> Result<Vector<Element<A>>,Error>{
        //Probably TODO: check to see if any elements of our vector
        //are out of bounds, the current bounds check is super limited
        //
        //Check to make sure the the length matches up with row size
        if other.len() > self.num_columns() {
            return Err(Error::SizeMismatch);
        }
        //Otherwise get on with the multiplication
        let mut output = Vec::new();
        
        let row_ranges = self.rows
            .iter()
            .zip(self.rows.iter().skip(1));
        for (row,(&start,&end)) in row_ranges.enumerate() {
            let mut sum: A = Default::default();
            for (column,&data) in  (1..).zip(self.data[start..end].iter()) {
                //We have to traverse to find the right element
                if let Some(Element(_,_,other_data)) = other.iter()
                    .filter(|Element(y,_,_)| *y==column)
                    .next()
                {

                    sum = sum + data**other_data;
                }

            }
            //Check to see if we should push a value
            if sum != Default::default() {
                //Assume we're all in the same column
                let &Element(_,column,_) = other.get(0).ok_or(Error::MalformedInput)?;
                //Here we create the now element of our sparse vector. 
                output.push(Element(row+1,column,sum)); 
            }
                

        }
        Ok(output)
    }
    pub fn vec_mul(&self, other: &Vector<A>) -> Result<Vector<A>,Error>{
        //Check to make sure the the length matches up with row size
        if other.len() != self.num_columns() {
            return Err(Error::SizeMismatch);
        }
        //Otherwise get on with the multiplication
        let mut output: Vec<A> = Vec::with_capacity(self.num_rows());
        //iter::repeat( Default::default()).take(self.num_rows).collect();
        
        let row_ranges = self.rows
            .iter()
            .zip(self.rows.iter().skip(1));
        for (&start,&end) in row_ranges {
            let mut sum: A = Default::default();
            for (i,&data) in  (start..end).zip(self.data[start..end].iter()) {
                sum = sum + data*other[self.columns[i] - 1];
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
        max_search_directions: usize
    ) -> Result<Vector<f64>,Error> {
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
        let mut x: Vector<f64> = [0.0f64].into_iter().cycle()
                                         .take(self.num_columns())
                                         // Why do I need a cloned?
                                         .cloned()
                                         .collect();
        let mut r = b.sub(&(self*&x));
        let r_norm = r.norm();
        let final_norm = r_norm * tolerance;

        // Our initial search direction, the first column of P
        let p = r.normalize();
        let mut big_b = DenseMatrix::new(vec![self * &p]);
        let mut big_p = DenseMatrix::new(vec![p]);
        loop {
            let alpha = big_b.least_squares(&r)?;


            x = x.add(&big_p.vec_mul(&alpha)?);

            r = r.sub(&big_b.vec_mul(&alpha)?);
            let norm = r.norm();
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
            }

        }


    }
}


// Multiplication by a scalar
//impl<A: Mul<Output=A> + Copy + Sized + Send + Sync> Mul<A> for Matrix<A> {
impl<A: Mul<Output=A> + MatrixElement> Mul<A> for Matrix<A> {
    type Output = Self;
    fn mul(mut self, other: A) -> Self{
        //Multiply all of the data by other
        //Um this does a ton of copies
        //But it's probably okay if we're working with numbers
        //TODO: benchmark against nonparallel version
        self.data.par_iter_mut()
                 .for_each_with(other, |o,x| *x = o.clone()*x.clone());
        self
    }

}

// Multiplication by a vector
impl<A: Mul<Output=A> + Add<Output=A> + MatrixElement + Default> Mul<&Vector<A>> 
for &Matrix<A> {
    //Should this be an option or should it panic?
    type Output = Vector<A>;
    fn mul(self, other: &Vector<A>) -> Self::Output{
        //This just wraps around one of our safe functions
        self.vec_mul(other).unwrap()
    }

}

// Multiplication by a sparse vector
impl<A: Mul<Output=A> + Add<Output=A> + MatrixElement + Default> Mul<&Vector<Element<A>>> 
for &Matrix<A> {
    //Should this be an option or should it panic?
    type Output = Vector<Element<A>>;
    fn mul(self, other: &Vector<Element<A>>) -> Self::Output{
        self.sparse_vec_mul(other).unwrap()
    }

}

// Multiplication by another matrix
impl<A: Mul<Output=A> + Add<Output=A> + MatrixElement + Default> Mul<&Matrix<A>> 
for &Matrix<A> {
    //Should this be an option or should it panic?
    type Output = Matrix<A>;
    fn mul(self, other: &Matrix<A>) -> Self::Output{
        // This just caches out to our safe matrix multiplication
        // But we unwrap stuff to make it more ergonomic, and there's
        // safe calls if you don't want to crash
        self.safe_mul(other).unwrap()
        
    }

}

impl<A: MatrixElement + Display + Default> Display for Matrix<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let elements: Vec<A> = self.all_elements()
            .collect();
        //We want to print a row out at a time
        let chunks = elements.chunks(self.num_columns());
        for chunk in chunks {
            write!(f,"[")?;
            for (i,element) in chunk.iter().enumerate() {
                write!(f,"{}",element)?;
                //Print all except for the trailing comma
                if i != chunk.len() - 1 {
                    write!(f,",")?;
                }
            }
            writeln!(f,"]")?;
        }
        Ok(())
    }
}
