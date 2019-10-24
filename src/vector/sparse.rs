use crate::Element;
use crate::error::Error;
use std::ops::{Add, Mul, Sub};

/// Sparse vectors are index value pairs.
/// Manipulate sparse vectors at your own risk,
/// index values must stay in order.
pub struct SparseVec<A> {
    data: Vec<A>,
    indices: Vec<usize>,
    length: usize,
    default: A,
}

impl<A: Element> SparseVec<A> {
    fn new(data: Vec<A>, indices: Vec<usize>, length: usize) -> Result<SparseVec<A>, Error> {
        indices.iter().zip(indices.iter().skip(1)).for_each(|(i, j)| {
            if i > j { 
                return Err(Error::MalformedInput);
            } else if j >= length {
                return Err(Error::ElementOutOfBounds);
            }
        });

        let default = A::default();
        SparseVec {
            data,
            indices,
            length,
            default,
        }
    }

    fn new_unsorted(data: Vec<(usize, A)>, length: usize) -> SparseVec<A> {
    }

    fn new_unsafe(data: Vec<A>, indices: Vec<usize>, length: usize) -> SparseVec<A> {
        let default = A::default;
        let length = data.len();
        SparseVec {
            data,
            indices,
            length,
            default,
        }
    }
}

impl SparseVec<f64> {
    fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).fold(0.0, |x, y| x + y).sqrt()
    }

    fn normalize(&self) -> SparseVec<f64> {
        self.scale(1.0 / (self.norm()))
    }
}

impl<A: Element + Mul<Output = A> + Add<Output = A> + Sub<Output = A>> SparseVec<A> {
    fn add(&self, other: &Self) -> Self {
        let len1 = self.len();
        let len2 = other.len();
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
                data.push(other[j]);
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

        let len = self.length;
        let default = A::default();
        SparseVec {
            data,
            indices,
            len,
            default,
        }
    }

    fn sub(&self, other: &SparseVec<A>) -> SparseVec<A> {
        let len1 = self.len();
        let len2 = other.len();
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
                data.push(A::default() - other[j]);
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
            &other.data[j..].iter().zip(&other.indices[j..].iter()).for_each(|(val, i)| {
                data.push(A::default() - val);
                indices.push(i);
            });
        } else if i < len1 {
            &self.data[i..].iter().zip(&self.indices[i..].iter()).for_each(|(val, i)| {
                data.push(val);
                indices.push(i);
            });
        }

        let len = self.length;
        let default = A::default();
        SparseVec {
            data,
            indices,
            len,
            default,
        }
    }

    fn scale(&self, scale: A) -> SparseVec<A> {
        let data = self.data.iter().map(|&x| x * scale).collect();
        let indices = self.indices.clone();
        let len = self.length;
        let default = A::default();

        SparseVec {
            data,
            indices,
            len,
            default,
        }
    }

    fn inner(&self, other: &SparseVec<A>) -> A {
        let len1 = self.len();
        let len2 = other.len();
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

        data.sum()
    }
}
