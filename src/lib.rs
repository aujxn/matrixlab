/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/*!
matrixlab is a small linear algebra library for rust featuring storage
and manipulation of sparse/dense matrices/vectors.

## Features

- Multiplication, addition, subtraction of sparse/dense matrices with
sparse/dense matrices or vectors

- Different iterators over sparse and dense matrices

- gmres, least squares, and backsolving of linear systems

## Plans

- LU and QR decompisitions
- More linear algebra

## Examples

Sparse matrix construction
```
use matrixlab::error::Error;
use matrixlab::MatrixElement;
use matrixlab::matrix::sparse::SparseMatrix;

let data = vec![(0usize, 0usize, 12i64), (3, 5, 4), (2, 2, 3), (1, 4, 42)];

let elements: Vec<MatrixElement<i64>> = data
    .iter()
    .map(|(i, j, val)| MatrixElement(*i, *j, *val))
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
use matrixlab::MatrixElement;
use matrixlab::matrix::sparse::SparseMatrix;
use matrixlab::vector::dense::DenseVec;

let elements = vec![
    MatrixElement(0, 0, 2u64),
    MatrixElement(0, 1, 1),
    MatrixElement(1, 0, 3),
    MatrixElement(1, 1, 7),
    MatrixElement(2, 2, 11),
    ];

let mat = SparseMatrix::new(3, 3, elements).unwrap();
let vec = DenseVec::new(vec![7, 2, 1]);

let result: DenseVec<u64> = &mat * &vec;
let expected = DenseVec::new(vec![16, 35, 11]);

assert_eq!(result, expected);
```

Sparse matrix - sparse matrix multiplication
```
use matrixlab::error::Error;
use matrixlab::MatrixElement;
use matrixlab::matrix::sparse::SparseMatrix;
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
    .map(|((m, n), points)| {
        let elements = points
            .iter()
            .map(|(i, j, val)| MatrixElement(*i, *j, *val))
            .collect();
            SparseMatrix::new(*m, *n, elements).unwrap()
    })
    .collect();

let result: SparseMatrix<i64> = &matrices[0] * &matrices[1];

let expected = c
    .iter()
    .enumerate()
    .map(|(index, val)| {
        let i = index / 4;
        let j = index % 4;
        MatrixElement(i as usize, j as usize, *val)
    })
    .collect();
let expected = SparseMatrix::new(3, 4, expected).unwrap();

assert_eq!(expected, result);
```
*/

/// Holds the error type for matrixlab
pub mod error;

/// Sparse and dense matricies
pub mod matrix;

/// Sparse and dense vectors
pub mod vector;

use error::Error;
use matrix::sparse::SparseMatrix;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// These are traits every element needs to have
/// Numbers trivially fulfill this
pub trait Element: Copy + Sized + Send + Sync + PartialEq {}

impl<A: Copy + Sized + Send + Sync + PartialEq> Element for A {}

#[derive(Clone, Debug, PartialEq, Copy)]
/// A point with i and j coordinates, as well as some data.
pub struct MatrixElement<A: Element>(pub usize, pub usize, pub A);

impl<A: Element> MatrixElement<A> {
    /// i is the row and j is the column index. These start at 0.
    pub fn new(i: usize, j: usize, data: A) -> MatrixElement<A> {
        MatrixElement(i, j, data)
    }
}

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
    let mut entries: Vec<MatrixElement<f64>> = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let line = lines.next().ok_or(Error::InvalidFile)??;
        //Here we read in the row and column
        let (row, column, data) = read_line(line, &ty)?;
        match prop {
            Prop::General => {}
            // Push the symmetric entry, unless we're on the diagonal
            _ => {
                if column != row {
                    entries.push(MatrixElement(column - 1, row - 1, data))
                }
            }
        }
        entries.push(MatrixElement(row - 1, column - 1, data));
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
