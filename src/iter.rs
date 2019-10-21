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

use crate::matrix::sparse::{Element, Matrix};
use crate::matrix::MatrixElement;

/// MatrixIter iterates over a sparse matrix. This iteration happens in order,
/// starting at the beginning of the first row and moving to the right and down.
/// If an element exists in the sparse matrix at that location then it is returned,
/// otherwise a zero element is returned. Note that it's probably exponential time
/// to run this over the whole matrix as I've implemented it in a particularly poor
/// way.
//NOTE: This implementation could be vastly improved, get is fairly slow
pub struct MatrixIter<'a, A: MatrixElement> {
    matrix: &'a Matrix<A>,
    row: usize,
    column: usize,
}

impl<'a, A: MatrixElement> MatrixIter<'a, A> {
    pub fn new(matrix: &'a Matrix<A>) -> MatrixIter<'a, A> {
        MatrixIter {
            row: 0,
            column: 0,
            //Set up a default value we can return references to
            matrix,
        }
    }
}

impl<'a, A: MatrixElement + Default> Iterator for MatrixIter<'a, A> {
    type Item = Element<&'a A>;
    fn next(&mut self) -> Option<Self::Item> {
        //can't panic because of manual bounds check
        //NOTE: will iterator modifiers like skip() break this?
        let val = self.matrix.get(self.row, self.column).unwrap();

        //check if we are in bounds
        if self.row == self.matrix.num_rows() {
            None
        } else {
            if self.column == self.matrix.num_columns() {
                self.column = 0;
                self.row += 1;
            } else {
                self.column += 1;
            }
            //can't panic because of manual bounds check
            Some(Element(self.row, self.column, val))
        }
    }
}

impl<'a, A: MatrixElement + Default> ExactSizeIterator for MatrixIter<'a, A> {
    fn len(&self) -> usize {
        self.matrix.num_rows() * self.matrix.num_columns()
    }
}

/// An ElementsIter iterates over all nonzero values in a sparse matrix. This is much
/// faster than a MatrixIter and should be preferred in most situations.
pub struct ElementsIter<'a, A: MatrixElement> {
    matrix: &'a Matrix<A>,
    row: usize,
    counter: usize,
    next_row_start: usize,
}

impl<'a, A: MatrixElement> ElementsIter<'a, A> {
    pub fn new(matrix: &'a Matrix<A>) -> ElementsIter<'a, A> {
        let next_row_start;
        match matrix.get_rows().get(1) {
            Some(i) => next_row_start = *i,
            None => next_row_start = matrix.nnz(),
        }
        ElementsIter {
            matrix,
            row: 0,
            counter: 0,
            next_row_start,
        }
    }
}

impl<'a, A: MatrixElement> Iterator for ElementsIter<'a, A> {
    type Item = Element<&'a A>;
    fn next(&mut self) -> Option<Self::Item> {
        let data = self.matrix.get_data().get(self.counter)?;
        let column = self.matrix.get_columns().get(self.counter)?;

        //when counter hits next row start then we have reached a new row
        //loop until next row is larger to find row of current data
        while self.counter == self.next_row_start {
            self.row += 1;
            match self.matrix.get_rows().get(self.row + 1) {
                Some(i) => self.next_row_start = *i,
                //if last row then iterate through the rest of the data
                None => self.next_row_start = self.matrix.nnz(),
            }
        }

        self.counter += 1;
        Some(Element(self.row, *column, data))
    }
}

impl<'a, A: MatrixElement> ExactSizeIterator for ElementsIter<'a, A> {
    fn len(&self) -> usize {
        self.matrix.nnz()
    }
}
