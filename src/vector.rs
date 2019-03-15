//
//    matrixlab, a library for working with sparse matricies
//    Copyright (C) 2019 Waylon Cude
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
//    

use std::ops::{Add,Mul,Sub};
use crate::matrix::MatrixElement;

pub type Vector<A> = Vec<A>;

//This trait lets us stick extra methods onto a Vector
pub trait VectorTrait<A> {
    /// Adds two vectors together
    fn add(&self,other: &Vector<A>) -> Vector<A>;
    /// Subtracts a vector from another
    fn sub(&self,other: &Vector<A>) -> Vector<A>;
    /// Scales the current vector by a constant
    fn scale(&self,scale: A) -> Vector<A>;
    /// Takes the inner product of this vector and another
    fn inner(&self,other: &Vector<A>) -> A;
}
pub trait FloatVectorTrait<A> {
    /// Takes the norm of a vector
    fn norm(&self) -> A;
    /// Normalizes a vector
    fn normalize(&self) -> Vector<f64>;
}
impl FloatVectorTrait<f64> for Vector<f64> {
    fn norm(&self) -> f64 {
        self.iter()
            .map(|x| x*x)
            .fold(0.0,|x,y| x+y)
            .sqrt()
    }
    fn normalize(&self) -> Vector<f64> {
        self.scale(1.0/(self.norm()))
    }
}
impl<A: MatrixElement + Mul<Output=A> + Add<Output=A> + Sub<Output=A> + Default> 
VectorTrait<A> for Vector<A> {
    fn add(&self,other: &Vector<A>) -> Vector<A> {
        self.iter().zip(other.iter())
            .map(|(&x,&y)| x + y)
            .collect()
    }
    fn sub(&self,other: &Vector<A>) -> Vector<A> {
        self.iter().zip(other.iter())
            .map(|(&x,&y)| x - y)
            .collect()
    }
    fn scale(&self,scale: A) -> Vector<A> {
        self.iter()
            .map(|&x| x * scale)
            .collect()
    }
    fn inner(&self,other: &Vector<A>) -> A {
        self.iter().zip(other.iter())
            .map(|(&x,&y)| x*y)
            .fold(Default::default(),|x,y| x+y)
    }
}

