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
use matrix::sparse::{Element, Matrix};

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Takes in a matrix market format file and gives back the
/// resulting matrix. This will always return a matrix filled
/// with floating point values to make parsing simpler. Matricies
/// with complex values will error out.
pub fn from_file(filename: &Path) -> Result<Matrix<f64>, Error> {
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
    let mut entries: Vec<Element<f64>> = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let line = lines.next().ok_or(Error::InvalidFile)??;
        //Here we read in the row and column
        let (row, column, data) = read_line(line, &ty)?;
        match prop {
            Prop::General => {}
            // Push the symmetric entry, unless we're on the diagonal
            _ => {
                if column != row {
                    entries.push(Element(column, row, data))
                }
            }
        }
        entries.push(Element(row, column, data));
    }
    // And finally we create the new matrix

    Matrix::new(rows, columns, entries)
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
