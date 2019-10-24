use crate::Element;
use crate::error::Error;
use std::ops::{Add, Mul, Sub};

/// Dense vectors are just normal variables where the type
/// of the vector meets the trait requirements for elements
///
/// This type definition is so more methods can be added to
/// the standard vector type.
pub struct DenseVec<A> {
    data: Vec<A>,
}

impl<A: Element> DenseVec<A> {
    fn new(data: Vec<A>) -> Self {
        DenseVec {
            data
        }
    }
}

impl DenseVec<f64> {
    fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).fold(0.0, |x, y| x + y).sqrt()
    }
    
    fn normalize(&self) -> DenseVec<f64> {
        self.scale(1.0 / (self.norm()))
    }
}

impl<A: Element + Mul<Output = A> + Add<Output = A> + Sub<Output = A>> DenseVec<A> {
    fn add(&self, other: &DenseVec<A>) -> DenseVec<A> {
        self.iter()
            .zip(other.iter())
            .map(|(&x, &y)| x + y)
            .collect()
    }

    fn sub(&self, other: &DenseVec<A>) -> DenseVec<A> {
        self.iter()
            .zip(other.iter())
            .map(|(&x, &y)| x - y)
            .collect()
    }

    fn scale(&self, scale: A) -> DenseVec<A> {
        self.iter().map(|&x| x * scale).collect()
    }

    fn inner(&self, other: &DenseVec<A>) -> A {
        self.iter()
            .zip(other.iter())
            .map(|(&x, &y)| x * y)
            .fold(A::default(), |x, y| x + y)
    }
}
