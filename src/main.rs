use itertools::join;
use matrixlab::matrix::gmres::gmres;
use matrixlab::matrix::sparse::SparseMatrix;
use matrixlab::vector::dense::DenseVec;
use matrixlab::MatrixElement;
use rand::prelude::*;

fn main() {
    let mut matrix: Vec<MatrixElement<f64>> = vec![];
    let mut x = vec![];
    let n = 1000;

    let mut rng = rand::thread_rng();
    for i in 0..n {
        x.push(rng.gen_range(-10.0, 10.0));
        for j in 0..n {
            if i == j {
                if rng.gen() {
                    matrix.push(MatrixElement(i, j, rng.gen_range(70.0, 100.0)));
                } else {
                    matrix.push(MatrixElement(i, j, rng.gen_range(-100.0, -70.0)));
                }
            } else if rng.gen_range(0, 100) > 25 {
                matrix.push(MatrixElement(i, j, rng.gen_range(-5.0, 5.0)));
            }
        }
    }

    let matrix = SparseMatrix::new(n, n, matrix).unwrap();
    let x = DenseVec::new(x);

    let b = &matrix * &x;

    let result = gmres(matrix.clone(), b.clone(), 0.0000001, 1000000, 50).unwrap();
    assert_eq!(
        join(x.get_data().iter().map(|x| format!("{:.5}", x)), &","),
        join(result.get_data().iter().map(|x| format!("{:.5}", x)), &",")
    );
}
