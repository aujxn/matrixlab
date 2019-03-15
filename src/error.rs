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


/// This is the general error type for our library. It can represent any failed
/// operation on a matrix, including building a new matrix or a failed operation
/// like matrix multiplication
#[derive(Debug)]
pub enum Error {
    /// This is returned when an operation fails due to a mismatch between the
    /// number of rows or columns in a matrix. This will happen when you multiply
    /// a AxB matrix by a CxD matrix, where B != C.
    SizeMismatch,
    /// This is returned when we get an error due to input that doesn't represent
    /// a valid matrix or vector
    MalformedInput,
    /// This is returned when there is an element that is out of bounds of the
    /// matrix during creation of a matrix.
    ElementOutOfBounds,
    /// This is returned when there is an element that is at the same location
    /// as another element during creation of a matrix.
    DuplicateElements,
    /// This error is a container for an IO error, returned when we would otherwise
    /// get an error when reading from a file
    IOError(std::io::Error),
    InvalidFile,
    ExceededIterations

}
impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        use Error::*;
        match (self,other) {
            (SizeMismatch,SizeMismatch) => true,
            (MalformedInput,MalformedInput) => true,
            (ElementOutOfBounds,ElementOutOfBounds) => true,
            (DuplicateElements,DuplicateElements) => true,
            (IOError(_),IOError(_)) => true,
            _ => false
        }
    }
}
impl From<std::io::Error> for Error {
    fn from(other: std::io::Error) -> Self {
        Error::IOError(other)
    }
}
impl From<std::num::ParseFloatError> for Error {
    fn from(_other: std::num::ParseFloatError) -> Self {
        Error::InvalidFile
    }
}
