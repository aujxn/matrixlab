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

    /// Returned from ['get_mut'] when no element is found at the provided location
    /// ['get_mut'] method.get_mut
    NotFound,

    /// This error is a container for an IO error, returned when we would otherwise
    /// get an error when reading from a file
    IOError(std::io::Error),

    InvalidFile,

    /// This error contains our closest guess for GMRES
    ExceededIterations(Vec<f64>),
}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        use Error::*;
        match (self, other) {
            (SizeMismatch, SizeMismatch) => true,
            (MalformedInput, MalformedInput) => true,
            (ElementOutOfBounds, ElementOutOfBounds) => true,
            (DuplicateElements, DuplicateElements) => true,
            (NotFound, NotFound) => true,
            (IOError(_), IOError(_)) => true,
            (InvalidFile, InvalidFile) => true,
            (ExceededIterations(_), ExceededIterations(_)) => true,
            _ => false,
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
