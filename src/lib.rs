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

/*!

matrixlab is a small linear algebra library for rust featuring storage
and manipulation of sparse/dense matrices/vectors.

The plan for this library is to feature graph processing algorithms for
sparse networks in the form of adjacency matrices.

## Features

- Multiplication, addition, subtraction of sparse/dense matrices with
sparse/dense matrices or vectors

- Different iterators over sparse and dense matrices

- gmres, least squares, and backsolving of linear systems

## Plans

- More linear algebra

- Graph coarsening with modularity functional

- Graph embedding algorithms

- Basic ML algorithms

## Examples

Sparse matrix construction
```
use matrixlab::error::Error;
use matrixlab::matrix::sparse::{MatElement, SparseMatrix};

let data = vec![(0usize, 0usize, 12i64), (3, 5, 4), (2, 2, 3), (1, 4, 42)];

let elements: Vec<MatElement<i64>> = data
    .iter()
    .map(|(i, j, val)| MatElement(*i, *j, *val))
    .collect();

let matrix = SparseMatrix::new(4, 6, elements).unwrap();

assert_eq!(matrix.get(0, 0), Ok(&12));
assert_eq!(matrix.get(3, 5), Ok(&4));
assert_eq!(matrix.get(2, 2), Ok(&3));
assert_eq!(matrix.get(1, 4), Ok(&42));
```

Sparse matrix - dense vector multiplication
```
use matrixlab::error::Error;
use matrixlab::matrix::sparse::{MatElement, SparseMatrix};

let elements = vec![
    MatElement(0, 0, 2u64),
    MatElement(0, 1, 1),
    MatElement(1, 0, 3),
    MatElement(1, 1, 7),
    MatElement(2, 2, 11),
    ];

let mat = SparseMatrix::new(3, 3, elements).unwrap();
let vec = vec![MatElement(0, 0, 7), MatElement(1, 0, 2), MatElement(2, 0, 1)];

let result: Vec<MatElement<u64>> = &mat * &vec;
let expected = vec![MatElement(0, 0, 16), MatElement(1, 0, 35), MatElement(2, 0, 11)];

assert_eq!(result, expected);
```

Sparse matrix - sparse matrix multiplication
```
use matrixlab::error::Error;
use matrixlab::matrix::sparse::{MatElement, SparseMatrix};
/*  _____   _______     _________
 * |2 0 3| |1 2 0 1|   |11 7 12 2|
 * |1 1 0|x|2 0 2 0| = | 3 2  2 1|
 * |3 0 1| |3 1 4 0|   | 6 7  4 3|
 *  ¯¯¯¯¯   ¯¯¯¯¯¯¯     ¯¯¯¯¯¯¯¯¯
 */
let a = ((3usize, 3usize),
    vec![(0usize, 0usize, 2i64),
        (0, 2, 3), (1, 0, 1),
        (1, 1, 1), (2, 0, 3),
        (2, 2, 1)]);

let b = ((3usize, 4usize),
    vec![(0usize, 0usize, 1i64),
        (0, 1, 2), (0, 3, 1),
        (1, 0, 2), (1, 2, 2),
        (2, 0, 3), (2, 1, 1),
        (2, 2, 4)]);

let c = vec![11i64, 7, 12, 2, 3, 2, 2, 1, 6, 7, 4, 3];

let matrices: Vec<SparseMatrix<i64>> = vec![a, b]
    .iter()
    .map(|((n, m), points)| {
        let elements = points
            .iter()
            .map(|(i, j, val)| MatElement(*i, *j, *val))
            .collect();
            SparseMatrix::new(*n, *m, elements).unwrap()
    })
    .collect();

let result: SparseMatrix<i64> = &matrices[0] * &matrices[1];
let result: Vec<i64> = result.elements()
    .map(|(_, _, val)| *val)
    .collect();

assert_eq!(c, result);
```
*/

/// Holds the error type for matrixlab
pub mod error;

/// Sparse and dense matricies
pub mod matrix;

/// Iterators over sparse matricies
pub mod iter;

/// Sparse and dense vectors
pub mod vector;

#[cfg(test)]
mod test;

use error::Error;
use matrix::sparse::{SparseMatrix, MatElement};

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// These are traits every element needs to have
// Numbers trivially fulfill this
pub trait Element: Default + Copy + Sized + Send + Sync + PartialEq {}

impl<A: Default + Copy + Sized + Send + Sync + PartialEq> Element for A {}

/// Takes in a matrix market format file and gives back the
/// resulting matrix. This will always return a matrix filled
/// with floating point values to make parsing simpler. Matricies
/// with complex values will error out.
pub fn from_file(filename: &Path) -> Result<SparseMatrix<f64>, Error> {
    // Read the file in and set up a reader over the lines
    let f = File::open(filename)?;
    let reader = BufReader::new(f);
    let mut lines = reader.lines();

    // First line tells us what kind of matrix it is
    let first = lines.next().ok_or(Error::InvalidFile)??;
    let (ty, prop) = get_type_and_prop(&first)?;

    // next_line lives outside the loop so we can save the first
    // non-comment line
    let mut next_line: String;
    // Then we have some comments
    // So throw them all away
    loop {
        next_line = lines.next().ok_or(Error::InvalidFile)??;
        if !next_line.starts_with("%") {
            //We're done parsing comments
            break;
        }
    }

    let elements = next_line
        .split_whitespace()
        .map(|e| e.parse::<usize>().map_err(|_| Error::InvalidFile))
        .collect::<Result<Vec<usize>, Error>>()?;
    // If we don't have enough elements
    if elements.len() < 3 {
        return Err(Error::InvalidFile);
    }

    let (rows, columns, num_entries) = (elements[0], elements[1], elements[2]);

    // And finally the entries
    // All of these become floating point numbers
    // TODO: we probably could add support for multiple types
    let mut entries: Vec<MatElement<f64>> = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let line = lines.next().ok_or(Error::InvalidFile)??;
        //Here we read in the row and column
        let (row, column, data) = read_line(line, &ty)?;
        match prop {
            Prop::General => {}
            // Push the symmetric entry, unless we're on the diagonal
            _ => {
                if column != row {
                    entries.push(MatElement(column, row, data))
                }
            }
        }
        entries.push(MatElement(row, column, data));
    }
    // And finally we create the new matrix

    SparseMatrix::new(rows, columns, entries)
}

#[derive(PartialEq, Eq)]
enum Type {
    Real,
    Integer,
    Complex,
    Pattern,
}

fn word_to_type(s: &str) -> Option<Type> {
    Some(match s {
        "real" => Type::Real,
        "integer" => Type::Integer,
        "complex" => Type::Complex,
        "pattern" => Type::Pattern,
        _ => return None,
    })
}

#[derive(PartialEq, Eq)]
enum Prop {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

fn word_to_prop(s: &str) -> Option<Prop> {
    Some(match s {
        "general" => Prop::General,
        "symmetric" => Prop::Symmetric,
        "skew-symmetric" => Prop::SkewSymmetric,
        "hermitian" => Prop::Hermitian,
        _ => return None,
    })
}

fn get_type_and_prop(first: &String) -> Result<(Type, Prop), Error> {
    let mut words = first.split_whitespace();
    if "%%MatrixMarket" != words.next().ok_or(Error::InvalidFile)? {
        return Err(Error::InvalidFile);
    }
    // We only support sparse matricies, so check for those here
    if "matrix" != words.next().ok_or(Error::InvalidFile)? {
        return Err(Error::InvalidFile);
    }
    if "coordinate" != words.next().ok_or(Error::InvalidFile)? {
        return Err(Error::InvalidFile);
    }
    let ty = word_to_type(words.next().ok_or(Error::InvalidFile)?).ok_or(Error::InvalidFile)?;
    let prop = word_to_prop(words.next().ok_or(Error::InvalidFile)?).ok_or(Error::InvalidFile)?;
    Ok((ty, prop))
}

fn read_line(line: String, ty: &Type) -> Result<(usize, usize, f64), Error> {
    //We probably could probably do some code reuse but for now
    //it's mostly copy + paste
    let words = line.split_whitespace();
    let elements = words
        .clone()
        .take(2)
        .map(|e| e.parse::<usize>().map_err(|_| Error::InvalidFile))
        .collect::<Result<Vec<usize>, Error>>()?;
    // If we don't have enough elements
    if elements.len() < 2 {
        return Err(Error::InvalidFile);
    }

    let (row, column) = (elements[0], elements[1]);

    let data: f64 = match ty {
        Type::Pattern => 1.0,
        _ => {
            let word = words.skip(2).next().ok_or(Error::InvalidFile)?;
            word.parse::<f64>()?
        }
    };
    Ok((row, column, data))
}
