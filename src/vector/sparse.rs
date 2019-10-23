
use crate::Element;
use crate::vector::{FloatVectorTrait, VectorTrait};
use std::ops::{Add, Mul, Sub};

/// Sparse vectors are index value pairs.
/// Manipulate sparse vectors at your own risk,
/// index values must stay in order.
pub type SparseVector<A> = Vec<(usize, A)>;


impl FloatVectorTrait<f64> for SparseVector<f64> {
    fn norm(&self) -> f64 {
    }
    fn normalize(&self) -> SparseVector<f64> {
    }
}

impl<A: Element + Mul<Output = A> + Add<Output = A> + Sub<Output = A>>
    VectorTrait<A> for SparseVector<A>
{
    fn add(&self, other: &SparseVector<A>) -> SparseVector<A> {
    }

    fn sub(&self, other: &SparseVector<A>) -> SparseVector<A> {
    }

    fn scale(&self, scale: A) -> SparseVector<A> {
    }

    fn inner(&self, other: &SparseVector<A>) -> A {
    }
}
