use matrixlab::matrix::sparse::SparseMatrix;
use matrixlab::MatrixElement;
use matrixlab::vector::dense::DenseVec;
use matrixlab::matrix::gmres::gmres;
use matrixlab::from_file;
use std::path::Path;

fn main() {

    let path = Path::new("mcca.mtx");
    let a = from_file(path).unwrap();
    let mut b = Vec::with_capacity(180);

    for i in 0..180{
        b.push(1 as f64)
    }
    let x = DenseVec::new(b);

    let b = &a * &x;

    let result = gmres(a, b, 1000.0, 1000000, 25).unwrap();
    print!("{:?}", result);
}

