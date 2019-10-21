/// Dense matricies
pub mod dense;

/// Sparse matricies
pub mod sparse;

// These are traits every element needs to have
// Numbers trivially fulfill this
pub trait MatrixElement: Default + Copy + Sized + Send + Sync + PartialEq {}

impl<A: Default + Copy + Sized + Send + Sync + PartialEq> MatrixElement for A {}
