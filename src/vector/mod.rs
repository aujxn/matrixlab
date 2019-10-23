/// Dense vectors
pub mod dense;

/// Sparse vectors
pub mod sparse;

pub type Vector<A> = Vec<A>;

//This trait lets us stick extra methods onto a Vector
pub trait VectorTrait<A> {
    /// Adds two vectors together
    fn add(&self, other: &Vector<A>) -> Vector<A>;
    /// Subtracts a vector from another
    fn sub(&self, other: &Vector<A>) -> Vector<A>;
    /// Scales the current vector by a constant
    fn scale(&self, scale: A) -> Vector<A>;
    /// Takes the inner product of this vector and another
    fn inner(&self, other: &Vector<A>) -> A;
}

pub trait FloatVectorTrait<A> {
    /// Takes the norm of a vector
    fn norm(&self) -> A;
    /// Normalizes a vector
    fn normalize(&self) -> Vector<f64>;
}

