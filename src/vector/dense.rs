
use crate::Element;
use crate::vector::{FloatVectorTrait, VectorTrait};
use std::ops::{Add, Mul, Sub};

/// Dense vectors are just normal variables where the type
/// of the vector meets the trait requirements for elements
///
/// This type definition is so more methods can be added to
/// the standard vector type.
pub type DenseVector<A> = Vec<A>;

impl FloatVectorTrait<f64> for DenseVector<f64> {
    fn norm(&self) -> f64 {
        self.iter().map(|x| x * x).fold(0.0, |x, y| x + y).sqrt()
    }
    fn normalize(&self) -> DenseVector<f64> {
        self.scale(1.0 / (self.norm()))
    }
}

impl<A: Element + Mul<Output = A> + Add<Output = A> + Sub<Output = A>>
    VectorTrait<A> for DenseVector<A>
{
    fn add(&self, other: &DenseVector<A>) -> DenseVector<A> {
        self.iter()
            .zip(other.iter())
            .map(|(&x, &y)| x + y)
            .collect()
    }

    fn sub(&self, other: &DenseVector<A>) -> DenseVector<A> {
        self.iter()
            .zip(other.iter())
            .map(|(&x, &y)| x - y)
            .collect()
    }

    fn scale(&self, scale: A) -> DenseVector<A> {
        self.iter().map(|&x| x * scale).collect()
    }

    fn inner(&self, other: &DenseVector<A>) -> A {
        self.iter()
            .zip(other.iter())
            .map(|(&x, &y)| x * y)
            .fold(A::default(), |x, y| x + y)
    }
}
