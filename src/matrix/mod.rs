pub mod dense;
pub mod sparse;

// These are traits every element needs to have
// Numbers trivially fulfill this
pub trait MatrixElement: Copy + Sized + Send + Sync + PartialEq {}

impl<A: Copy + Sized + Send + Sync + PartialEq> MatrixElement for A {}

