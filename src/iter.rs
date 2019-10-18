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

use crate::matrix::sparse::{Matrix,Element};
use crate::matrix::MatrixElement;

/// MatrixIter iterates over a sparse matrix. This iteration happens in order,
/// starting at the beginning of the first row and moving to the right and down.
/// If an element exists in the sparse matrix at that location then it is returned,
/// otherwise a zero element is returned. Note that it's probably exponential time
/// to run this over the whole matrix as I've implemented it in a particularly poor
/// way.
//NOTE: This implementation could be vastly improved, get is fairly slow
pub struct MatrixIter<'a,A: MatrixElement> {
    matrix: &'a Matrix<A>,
    row: usize,
    column: usize,

}
impl<'a,A: MatrixElement > MatrixIter<'a,A> {
    pub fn new(matrix: &'a Matrix<A>) -> MatrixIter<'a,A> {
        MatrixIter {
            row: 1,
            column: 1,
            //Set up a default value we can return references to
            matrix
        }
    }
}
impl<'a,A: MatrixElement + Default> Iterator for MatrixIter<'a,A> {
    type Item = A;
    // We can easily calculate the remaining number of iterations.
    fn next(&mut self) -> Option<Self::Item> {

        let result = self.matrix.get(self.row,self.column);


        self.column += 1;
        //If we go past the last row
        if self.row > self.matrix.num_rows() {
            //Then stop iterating
            return None;
        //If we go past the end of the row
        } else if self.column > self.matrix.num_columns() {
            //Reset the columns
            self.column = 1;
            //And increment the rows
            self.row += 1;
        }
        //TODO: clone bad
        Some(match result {
            Some(x) => x.clone(),
            None => Default::default()
        })

    }
}
impl<'a,A: MatrixElement + Default> ExactSizeIterator for MatrixIter<'a,A> {
    // We can easily calculate the remaining number of iterations.
    fn len(&self) -> usize {
        self.matrix.num_rows()*self.matrix.num_columns()
    }
}

/// An ElementsIter iterates over all nonzero values in a sparse matrix. This is much
/// faster than a MatrixIter and should be preferred in most situations.
pub struct ElementsIter<'a,A: MatrixElement> {
    matrix: &'a Matrix<A>,
    row: usize,
    counter: usize

}
impl<'a,A: MatrixElement> ElementsIter<'a,A> {
    pub fn new(matrix: &'a Matrix<A>) -> ElementsIter<'a,A> {
        ElementsIter {
            matrix,
            row: 0,
            counter: 0
        }
    }

}
impl<'a,A: MatrixElement> Iterator for ElementsIter<'a,A> {
    //Should this return a whole point or a reference?
    type Item = Element<A>;
    // We can easily calculate the remaining number of iterations.
    fn next(&mut self) -> Option<Self::Item> {
        let data = self.matrix.get_data().get(self.counter)?;
        let column = self.matrix.get_columns().get(self.counter)?;
        
        loop {
            let next_row_start = self.matrix.get_rows().get(self.row+1)?;
            if self.counter >= *next_row_start {
                self.row += 1; 
            } else {
                break;
            }
        }
        
        let row = self.row + 1;
        self.counter += 1;
        Some(Element(row,*column,*data))
    }
}
impl<'a,A: MatrixElement> ExactSizeIterator for ElementsIter<'a,A> {
    // We can easily calculate the remaining number of iterations.
    fn len(&self) -> usize {
        self.matrix.number_of_nonzero()
    }
}
