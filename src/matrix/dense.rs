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

use std::ops::{Mul,Add,Sub};
use crate::matrix::MatrixElement;
use crate::vector::{Vector,VectorTrait,FloatVectorTrait};
use crate::error::Error;
use rayon::prelude::*;

#[derive(PartialEq,Debug)]
pub struct DenseMatrix<A> {
    columns: Vec<Vector<A>>
}
impl<A> DenseMatrix<A> {
    pub fn new(columns: Vec<Vector<A>>) -> DenseMatrix<A> {
        DenseMatrix {
            columns
        }
    }
    pub fn add_column(&mut self, column: Vector<A>) {
        self.columns.push(column);
    }
    pub fn num_rows(&self) -> usize {
        // We assume the matrix is well formed so we just look
        // at the number of rows in the first column
        match self.columns.get(0) {
            Some(column) => column.len(),
            None => 0
        }
    }
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

}
//Maybe todo: should this be copy?
impl<A: Clone> DenseMatrix<A> {
    pub fn transpose(&self) -> DenseMatrix<A> {
        // Set up the columns for our new matrix
        let mut columns = Vec::with_capacity(self.num_rows());
        for _ in 0 .. self.num_rows() {
            columns.push(Vec::with_capacity(self.num_columns()));
        }

        // Set up the elements of the columns of our new array
        for column in self.columns.iter() {
            for (i,entry) in column.iter().enumerate() {
                columns[i].push(entry.clone());
            }
        }

        DenseMatrix::new(columns)
    }
}

//TODO: Make this generic?
impl<A: MatrixElement + Mul<Output=A> + Add<Output=A> + Sub<Output=A> + Default> DenseMatrix<A> {
    pub fn vec_mul(&self, other: &Vector<A>) -> Vector<A> {
        self.columns.par_iter()
            .zip(other.par_iter())
            // Is there any way to make this a normal iterator
            // and still be able to flatten?
            // Is collecting it slow?
            .map(|(column,scale)| column.iter()
                                        .map(|x| *x**scale)
                                        .collect::<Vec<A>>())
            .reduce(
                || [Default::default()]
                   .into_iter()
                   .cycle()
                   .take(self.num_rows())
                   .cloned()
                   .collect(),
                |x,y| x.add(&y))
            //.collect()
    }
    pub fn safe_mul(&self, other: &DenseMatrix<A>) -> Result<DenseMatrix<A>,Error> {
        if self.num_columns() != other.num_rows() {
            return Err(Error::SizeMismatch);
        }

        let new_cols = other.columns.par_iter()
            .map_with(self, |&mut s, col| s.vec_mul(col))
            .collect();

        Ok(DenseMatrix::new(new_cols))
        
    }
}
impl DenseMatrix<f64> {
    /// This takes an upper triangular matrix, and solves it to
    /// equal b
    pub fn backsolve(&self, b: &Vector<f64>) -> Vector<f64> {
        // Start off with a copy of b, to modify to create our solutions
        let mut solutions: Vec<f64> = b.clone();
        // Start with the last column
        for (i,column) in self.columns.iter().rev().enumerate() {
            //Normalize our last element
            let last_element = b.len()-1-i;
            solutions[last_element] /= column[last_element];
            //And skip i elements because they're all zero
            //But we have to reverse the list first
            for (j,element) in column.iter().rev().skip(1+i).enumerate() {
                //Move up b as we iterate
                let last_element = b.len()-1-i;
                //And move up b as we go up each column
                //This probably won't overflow
                //TODO ^ figure out if this is exploitable
                let current_element = last_element - 1-j;
                solutions[current_element] -= solutions[last_element] * element;                                            
            }
        }
        solutions
    }
    /// This solves for B*y = r
    pub fn least_squares(&self, r: &Vector<f64>) -> Vector<f64> {
        //Solve for Q, for our QR factorization
        let q = self.factor_q();
        println!("Q: {:?}",q);
        let q_transpose = q.transpose();
        println!("q_transpose: {:?}",q_transpose);
        let rhs = q_transpose.vec_mul(&r);
        println!("rhs: {:?}",rhs);
        let r = q_transpose.safe_mul(self).expect("Error in least squares, multiplication failed");;
        println!("R: {:?}",r);

        //Now solve for Ra = rhs, then return a
        r.backsolve(&rhs)
    }
    pub fn factor_q(&self) -> DenseMatrix<f64> {
        let mut q_vectors: Vec<Vector<f64>> = Vec::with_capacity(self.num_columns());
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
    pub fn orthogonal_to(&self,other: &Vector<f64>) -> Vector<f64> {
        let mut final_vec = other.clone();
        for column in self.columns.iter() {
            final_vec = final_vec.sub(&column.scale(other.inner(column)));
        }
        final_vec
    }
}
