/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use crate::matrix::sparse::SparseMatrix;
use crate::Element;

/// MatrixIter iterates over a sparse matrix. This iteration happens in order,
/// starting at the beginning of the first row and moving to the right and down.
/// If an element exists in the sparse matrix at that location then it is returned,
/// otherwise a zero element is returned. Note that it's probably exponential time
/// to run this over the whole matrix as I've implemented it in a particularly poor
/// way.
//NOTE: This implementation could be vastly improved, get is fairly slow
pub struct MatrixIter<'a, A: Element> {
    matrix: &'a SparseMatrix<A>,
    row: usize,
    column: usize,
}

impl<'a, A: Element> MatrixIter<'a, A> {
    pub fn new(matrix: &'a SparseMatrix<A>) -> MatrixIter<'a, A> {
        MatrixIter {
            row: 0,
            column: 0,
            //Set up a default value we can return references to
            matrix,
        }
    }
}

impl<'a, A: Element> Iterator for MatrixIter<'a, A> {
    type Item = (usize, usize, Option<&'a A>);
    fn next(&mut self) -> Option<Self::Item> {
        //check if we are in bounds
        if self.row >= self.matrix.num_rows() {
            None
        } else {
            let val = self.matrix.get(self.row, self.column);
            let row = self.row;
            let col = self.column;
            if self.column == self.matrix.num_columns() - 1 {
                self.column = 0;
                self.row += 1;
            } else {
                self.column += 1;
            }

            match val {
                Ok(val) => Some((row, col, Some(val))),
                Err(_) => Some((row, col, None)),
            }
        }
    }
}

impl<'a, A: Element> ExactSizeIterator for MatrixIter<'a, A> {
    fn len(&self) -> usize {
        self.matrix.num_rows() * self.matrix.num_columns()
    }
}

/// An ElementsIter iterates over all nonzero values in a sparse matrix. This is much
/// faster than a MatrixIter and should be preferred in most situations.
pub struct ElementsIter<'a, A: Element> {
    matrix: &'a SparseMatrix<A>,
    row: usize,
    counter: usize,
    next_row_start: usize,
}

impl<'a, A: Element> ElementsIter<'a, A> {
    pub fn new(matrix: &'a SparseMatrix<A>) -> ElementsIter<'a, A> {
        let next_row_start;
        match matrix.get_rows().get(1) {
            Some(i) => next_row_start = *i,
            None => next_row_start = matrix.num_nonzero(),
        }
        ElementsIter {
            matrix,
            row: 0,
            counter: 0,
            next_row_start,
        }
    }
}

impl<'a, A: Element> Iterator for ElementsIter<'a, A> {
    type Item = (usize, usize, &'a A);
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
                None => self.next_row_start = self.matrix.num_nonzero(),
            }
        }

        self.counter += 1;
        Some((self.row, *column, data))
    }
}

impl<'a, A: Element> ExactSizeIterator for ElementsIter<'a, A> {
    fn len(&self) -> usize {
        self.matrix.num_nonzero()
    }
}

pub struct RowIter<'a, A: Element> {
    matrix: &'a SparseMatrix<A>,
    row: usize,
}

impl<'a, A: Element> RowIter<'a, A> {
    pub fn new(matrix: &'a SparseMatrix<A>) -> RowIter<'a, A> {
        RowIter { matrix, row: 0 }
    }
}

impl<'a, A: Element> Iterator for RowIter<'a, A> {
    type Item = (&'a [usize], &'a [A]);
    fn next(&mut self) -> Option<Self::Item> {
        if self.row == self.matrix.num_rows() {
            None
        } else {
            let row_start = self.matrix.get_rows().get(self.row).unwrap();
            let row_end = self.matrix.get_rows().get(self.row + 1).unwrap();
            let data = self.matrix.get_data();
            let cols = self.matrix.get_columns();

            self.row += 1;
            Some((&cols[*row_start..*row_end], &data[*row_start..*row_end]))
        }
    }
}

impl<'a, A: Element> ExactSizeIterator for RowIter<'a, A> {
    fn len(&self) -> usize {
        self.matrix.num_rows()
    }
}
