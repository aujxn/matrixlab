use crate::Element;
use super::sparse::SparseVec;
use std::ops::{Add, Mul, Sub, SubAssign};

/// Dense vectors are just normal variables where the type
/// of the vector meets the trait requirements for elements
///
/// This type definition is so more methods can be added to
/// the standard vector type.
#[derive(PartialEq, Debug)]
pub struct DenseVec<A> {
    data: Vec<A>,
}

/// Constructor for a dense vector.
impl<A: Element> DenseVec<A> {
    pub fn new(data: Vec<A>) -> Self {
        DenseVec {
            data
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> std::slice::Iter<A> {
        self.data.iter()
    }

    pub fn get_data(&self) -> &Vec<A> {
        &self.data
    }
}

//TODO: update to work with complex and f32 types
impl DenseVec<f64> {
    /// Calculates the Euclidean norm of a dense vector.
    /// This is the magnitude of the vector.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).fold(0.0, |x, y| x + y).sqrt()
    }
    
    /// Evaluates the unit vector in the same direction.
    pub fn normalize(&self) -> Self {
        self.scale(1.0 / (self.norm()))
    }
}

impl<A: Element + Add<Output = A>> DenseVec<A> {
    /// Adds two dense vectors by adding elements with the same index.
    pub fn add(&self, other: &Self) -> Self {
        let data = self.data.iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        DenseVec::new(data)
    }

    /// Adds a dense vector and a sparse vector.
    pub fn add_sparse(&self, other: &SparseVec<A>) -> Self {
        let mut sparse_iter = other.get_data().iter();
        let mut data = Vec::with_capacity(self.data.len());

        let mut current = sparse_iter.next();

        for (i, &val) in self.data.iter().enumerate() {
            match current {
                Some(&(j, sparse_val)) => {
                    if i < j {
                        data.push(val);
                    } else if i == j {
                        data.push(val + sparse_val);
                        current = sparse_iter.next();
                    }
                },
                None => data.push(val),
            }
        }
        DenseVec::new(data)
    }
}

impl<A: Element + SubAssign> DenseVec<A> {
    pub fn sub_mut(&mut self, other: &Self) {
        self.data.iter_mut().zip(other.data.iter()).for_each(|(&mut i, &j)| i -= j);
    }
}

impl<A: Element + Mul<Output = A> + Add<Output = A> + Sub<Output = A>> DenseVec<A> {
    /// Subtracts a dense vector from a dense vector.
    pub fn sub(&self, other: &Self) -> Self {
        let data = self.data.iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x - y)
            .collect();
        DenseVec::new(data)
    }

    /// Subtracts a sparse vector from a dense vector.
    pub fn sub_sparse(&self, other: &SparseVec<A>) -> Self {
        let mut sparse_iter = other.get_data().iter();
        let mut data = Vec::with_capacity(self.data.len());

        let mut current = sparse_iter.next();

        for (i, &val) in self.data.iter().enumerate() {
            match current {
                Some(&(j, sparse_val)) => {
                    if i < j {
                        data.push(val);
                    } else if i == j {
                        data.push(val - sparse_val);
                        current = sparse_iter.next();
                    }
                },
                None => data.push(val),
            }
        }
        DenseVec::new(data)
    }
}

impl<A: Element + Mul<Output = A>> DenseVec<A> {
    /// Multiplies every element in the vector by a scalar.
    pub fn scale(&self, scale: A) -> Self {
        let data = self.data.iter().map(|&x| x * scale).collect();
        DenseVec::new(data)
    }
}

impl<A: Element + Mul<Output = A> + Add<Output = A> + Default> DenseVec<A> {
    /// Calculates the inner product of two dense vectors.
    /// This is the dot product.
    pub fn inner(&self, other: &Self) -> A {
        self.data.iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x * y)
            .fold(A::default(), |x, y| x + y)
    }

    /// Calculates the inner product of a dense and sparse vector.
    /// This is the dot product.
    pub fn inner_sparse(&self, other: &SparseVec<A>) -> A {
        other.get_data().iter().map(|&(i, val)| val * self.data[i]).fold(A::default(), |acc, prod| acc + prod)
    }
}
