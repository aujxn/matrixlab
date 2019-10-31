/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use crate::Element;
use super::dense::DenseVec;
use crate::error::Error;
use std::ops::{Add, Mul, Sub};
use rayon::prelude::*;

/// Sparse vectors are index value pairs.
/// Manipulate sparse vectors at your own risk,
/// index values must stay in order.
pub struct SparseVec<A> {
    data: Vec<(usize, A)>,
    // the "actual" length of the vector (including 0's)
    length: usize,
}

impl<A: Element> SparseVec<A> {
    /// Constructs a sparse vector. Checks to make sure that the indices
    /// provided are in order and in range. The error types are MalformedInput
    /// and ElementOutOfBounds accordingly.
    pub fn new(data: Vec<(usize, A)>, length: usize) -> Result<Self, Error> {
        for ((i, _), (j, _)) in data.iter().zip(data.iter().skip(1)) {
            if *i >= *j { 
                return Err(Error::MalformedInput)
            } else if *j >= length {
                return Err(Error::ElementOutOfBounds)
            }
        }

        Ok(SparseVec {
            data,
            length,
        })
    }

    /// Constructs a sparse matrix from a Vec of index, value pairs.
    /// This method sorts the data for you and should be avoided
    /// if your data is alreaded sorted.
    ///
    /// Possible error type is ElementOutOfBounds.
    pub fn new_unsorted(mut data: Vec<(usize, A)>, length: usize) -> Result<Self, Error> {
        data.par_sort_unstable_by(|&(i1, _), &(i2, _)| i1.cmp(&i2));

        for ((i, _), (j, _)) in data.iter().zip(data.iter().skip(1)) {
            if *i >= *j { 
                return Err(Error::MalformedInput)
            } else if *j >= length {
                return Err(Error::ElementOutOfBounds)
            }
        }

        Ok(SparseVec {
            data,
            length,
        })
    }

    /// Constructs a sparse matrix from values and indices provided.
    /// Performs no validity checks.
    pub fn new_unsafe(data: Vec<(usize, A)>, length: usize) -> Self {
        SparseVec {
            data,
            length,
        }
    }

    /// Returns the "true" length of the vector
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns the number of non-zero elements
    pub fn num_nonzero(&self) -> usize {
        self.data.len()
    }

    /// Gets a reference to the underlying data
    pub fn get_data(&self) -> &Vec<(usize, A)> {
        &self.data
    }
}

//TODO: implement num traits so this is less limiting
impl SparseVec<f64> {
    /// Calculates the Euclidean norm of a sparse vector.
    /// This is the magnitude of the vector.
    fn norm(&self) -> f64 {
        self.data.iter().map(|(_, x)| x * x).fold(0.0, |acc, y| acc + y).sqrt()
    }

    /// Evaluates the unit vector in the same direction.
    fn normalize(&self) -> Self {
        self.scale(1.0 / (self.norm()))
    }
}

impl<A: Element + Sub<Output = A> + Default> SparseVec<A> {
    /// Subtracts a sparse vector from a sparse vector.
    fn sub(&self, other: &Self) -> Self {
        let len1 = self.data.len();
        let len2 = other.data.len();
        let mut i = 0;
        let mut j = 0;

        //too much capacity?
        let mut data = Vec::with_capacity(len1 + len2);

        while i < len1 && j < len2 {
            let index1 = self.data[i].0;
            let index2 = other.data[j].0;
            if index1 < index2 {
                data.push(self.data[i]);
                i += 1;
            } else if index1 > index2 {
                data.push((index2, A::default() - other.data[j].1));
                j += 1;
            } else {
                let diff = self.data[i].1 - other.data[j].1;
                data.push((index1, diff));
                i += 1;
                j += 1;
            }
        }

        if i == len1 && j < len2 {
            other.data[j..].into_iter().for_each(|(i, val)| {
                data.push((*i, A::default() - *val));
            });
        } else if i < len1 {
            self.data[i..].into_iter().for_each(|(i, val)| {
                data.push((*i, *val));
            });
        }

        let length = self.length;
        SparseVec {
            data,
            length,
        }
    }

    /// Subtracts a dense vector from a sparse vector.
    /// Returns a dense vector because the result is likely dense.
    fn sub_dense(&self, other: &DenseVec<A>) -> DenseVec<A> {
        let mut data = Vec::with_capacity(self.length);
        let mut sparse_iter = self.data.iter();
        let mut current = sparse_iter.next();

        for (i, &val) in other.get_data().iter().enumerate() {
            match current {
                Some(&(j, sparse_val)) => {
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
}

impl<A: Element + Add<Output = A>> SparseVec<A> {
    /// Adds two sparse vectors by adding elements with the same index.
    // TODO: make safe?
    fn add(&self, other: &Self) -> Self {
        let len1 = self.data.len();
        let len2 = other.data.len();
        let mut i = 0;
        let mut j = 0;

        //too much capacity?
        let mut data = Vec::with_capacity(len1 + len2);

        while i < len1 && j < len2 {
            let index1 = self.data[i].0;
            let index2 = other.data[j].0;
            if index1 < index2 {
                data.push(self.data[i]);
                i += 1;
            } else if index1 > index2 {
                data.push(other.data[j]);
                j += 1;
            } else {
                let sum = self.data[i].1 + other.data[j].1;
                data.push((index1, sum));
                i += 1;
                j += 1;
            }
        }

        if i == len1 && j < len2 {
            data.extend_from_slice(&other.data[j..]);
        } else if i < len1 {
            data.extend_from_slice(&self.data[i..]);
        }

        let length = self.length;
        SparseVec {
            data,
            length,
        }
    }

    /// Adds a sparse vector and a dense vector.
    /// This returns a dense vector because the result
    /// is likely more dense than the arguments
    fn add_dense(&self, other: &DenseVec<A>) -> DenseVec<A> {
        other.add_sparse(&self)
    }
}

impl<A: Element + Mul<Output = A>> SparseVec<A> {
    /// Multiplies every element in the vector by a scalar.
    fn scale(&self, scale: A) -> Self {
        let data = self.data.iter().map(|&(i, x)| (i, x * scale)).collect();
        let length = self.length;

        SparseVec {
            data,
            length,
        }
    }
}

impl<A: Element + Mul<Output = A> + Add<Output = A> + Default> SparseVec<A> {
    /// Calculates the inner product of two sparse vectors.
    /// This is the dot product.
    fn inner(&self, other: &Self) -> A {
        let len1 = self.data.len();
        let len2 = other.data.len();
        let mut i = 0;
        let mut j = 0;

        let mut data = Vec::with_capacity(len1);

        while i < len1 && j < len2 {
            let index1 = self.data[i].0;
            let index2 = other.data[j].0;
            if index1 < index2 {
                i += 1;
            } else if index1 > index2 {
                j += 1;
            } else {
                let prod = self.data[i].1 * other.data[j].1;
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
