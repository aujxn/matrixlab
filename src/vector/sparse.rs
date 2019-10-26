use crate::Element;
use super::dense::DenseVec;
use crate::error::Error;
use std::ops::{Add, Mul, Sub};
use rayon::prelude::*;

/// Sparse vectors are index value pairs.
/// Manipulate sparse vectors at your own risk,
/// index values must stay in order.
pub struct SparseVec<A> {
    pub data: Vec<A>,
    pub indices: Vec<usize>,
    pub length: usize,
    default: A,
}

impl<A: Element> SparseVec<A> {
    /// Constructs a sparse vector. Checks to make sure that the indices
    /// provided are in order and in range. The error types are MalformedInput
    /// and ElementOutOfBounds accordingly.
    pub fn new(data: Vec<A>, indices: Vec<usize>, length: usize) -> Result<Self, Error> {
        for (&i, &j) in indices.iter().zip(indices.iter().skip(1)) {
            if i >= j { 
                return Err(Error::MalformedInput)
            } else if j >= length {
                return Err(Error::ElementOutOfBounds)
            }
        }

        let default = A::default();
        Ok(SparseVec {
            data,
            indices,
            length,
            default,
        })
    }

    /// Constructs a sparse matrix from a Vec of index, value pairs.
    /// This method sorts the data for you and should be avoided
    /// if your data is alreaded sorted.
    ///
    /// Possible error type is ElementOutOfBounds.
    fn new_unsorted(mut data: Vec<(usize, A)>, length: usize) -> Result<Self, Error> {
        data.par_sort_unstable_by(|&(i1, _), &(i2, _)| i1.cmp(&i2));
        let (indices, data): (Vec<_>, Vec<_>) = data.iter().cloned().unzip();

        for (&i, &j) in indices.iter().zip(indices.iter().skip(1)) {
            if i >= j { 
                return Err(Error::MalformedInput)
            } else if j >= length {
                return Err(Error::ElementOutOfBounds)
            }
        }

        let default = A::default();
        Ok(SparseVec {
            data,
            indices,
            length,
            default,
        })
    }

    /// Constructs a sparse matrix from values and indices provided.
    /// Performs no validity checks.
    fn new_unsafe(data: Vec<A>, indices: Vec<usize>, length: usize) -> Self {
        let default = A::default();
        let length = data.len();
        SparseVec {
            data,
            indices,
            length,
            default,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl SparseVec<f64> {
    /// Calculates the Euclidean norm of a sparse vector.
    /// This is the magnitude of the vector.
    fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).fold(0.0, |x, y| x + y).sqrt()
    }

    /// Evaluates the unit vector in the same direction.
    fn normalize(&self) -> Self {
        self.scale(1.0 / (self.norm()))
    }
}

impl<A: Element + Mul<Output = A> + Add<Output = A> + Sub<Output = A>> SparseVec<A> {
    /// Adds two sparse vectors by adding elements with the same index.
    fn add(&self, other: &Self) -> Self {
        let len1 = self.data.len();
        let len2 = other.data.len();
        let mut i = 0;
        let mut j = 0;

        //too much capacity?
        let mut data = Vec::with_capacity(len1 + len2);
        let mut indices = Vec::with_capacity(len1 + len2);

        while i < len1 && j < len2 {
            let index1 = self.indices[i];
            let index2 = other.indices[j];
            if index1 < index2 {
                data.push(self.data[i]);
                indices.push(index1);
                i += 1;
            } else if index1 > index2 {
                data.push(other.data[j]);
                indices.push(index2);
                j += 1;
            } else {
                let sum = self.data[i] + other.data[j];
                data.push(sum);
                indices.push(index1);
                i += 1;
                j += 1;
            }
        }

        if i == len1 && j < len2 {
            data.extend_from_slice(&other.data[j..]);
            indices.extend_from_slice(&other.indices[j..]);
        } else if i < len1 {
            data.extend_from_slice(&self.data[i..]);
            indices.extend_from_slice(&self.indices[j..]);
        }

        let length = self.length;
        let default = A::default();
        SparseVec {
            data,
            indices,
            length,
            default,
        }
    }

    /// Adds a sparse vector and a dense vector.
    /// This returns a dense vector because the result
    /// is likely more dense than the arguments
    fn add_dense(&self, other: &DenseVec<A>) -> DenseVec<A> {
        other.add_sparse(&self)
    }

    /// Subtracts a sparse vector from a sparse vector.
    fn sub(&self, other: &Self) -> Self {
        let len1 = self.data.len();
        let len2 = other.data.len();
        let mut i = 0;
        let mut j = 0;

        //too much capacity?
        let mut data = Vec::with_capacity(len1 + len2);
        let mut indices = Vec::with_capacity(len1 + len2);

        while i < len1 && j < len2 {
            let index1 = self.indices[i];
            let index2 = other.indices[j];
            if index1 < index2 {
                data.push(self.data[i]);
                indices.push(index1);
                i += 1;
            } else if index1 > index2 {
                data.push(A::default() - other.data[j]);
                indices.push(index2);
                j += 1;
            } else {
                let diff = self.data[i] - other.data[j];
                data.push(diff);
                indices.push(index1);
                i += 1;
                j += 1;
            }
        }

        if i == len1 && j < len2 {
            other.data[j..].into_iter().zip(other.indices[j..].into_iter()).for_each(|(val, i)| {
                data.push(A::default() - *val);
                indices.push(*i);
            });
        } else if i < len1 {
            self.data[i..].into_iter().zip(self.indices[i..].into_iter()).for_each(|(val, i)| {
                data.push(*val);
                indices.push(*i);
            });
        }

        let length = self.length;
        let default = A::default();
        SparseVec {
            data,
            indices,
            length,
            default,
        }
    }

    /// Subtracts a dense vector from a sparse vector.
    /// Returns a dense vector because the result is likely dense.
    fn sub_dense(&self, other: &DenseVec<A>) -> DenseVec<A> {
        let mut data = Vec::with_capacity(self.length);
        let mut sparse_iter = self.indices.iter().zip(self.data.iter());
        let mut current = sparse_iter.next();

        for (i, &val) in other.data.iter().enumerate() {
            match current {
                Some((&j, &sparse_val)) => {
                    if i < j {
                        data.push(A::default() - val);
                    } else if i == j {
                        data.push(sparse_val - val);
                        current = sparse_iter.next();
                    }
                },
                None => data.push(A::default() - val),
            }
        }
        DenseVec::new(data)
    }

    /// Multiplies every element in the vector by a scalar.
    fn scale(&self, scale: A) -> Self {
        let data = self.data.iter().map(|&x| x * scale).collect();
        let indices = self.indices.clone();
        let length = self.length;
        let default = A::default();

        SparseVec {
            data,
            indices,
            length,
            default,
        }
    }

    /// Calculates the inner product of two sparse vectors.
    /// This is the dot product.
    fn inner(&self, other: &Self) -> A {
        let len1 = self.data.len();
        let len2 = other.data.len();
        let mut i = 0;
        let mut j = 0;

        let mut data = Vec::with_capacity(len1);

        while i < len1 && j < len2 {
            let index1 = self.indices[i];
            let index2 = other.indices[j];
            if index1 < index2 {
                i += 1;
            } else if index1 > index2 {
                j += 1;
            } else {
                let prod = self.data[i] * other.data[j];
                data.push(prod);
                i += 1;
                j += 1;
            }
        }

        data.iter().fold(A::default(), |sum, &val| sum + val)
    }

    /// Calculates the inner product of a sparse and dense vector.
    /// This is the dot product.
    fn inner_dense(&self, other: &DenseVec<A>) -> A {
        other.inner_sparse(&self)
    }
}
