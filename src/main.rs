use itertools::join;
use matrixlab::from_file;
use matrixlab::matrix::gmres::gmres;
use matrixlab::matrix::sparse::SparseMatrix;
use matrixlab::vector::dense::DenseVec;
use matrixlab::MatrixElement;
use rand::prelude::*;
use std::path::Path;

fn main() {
    let (x, result) = solve_file_random_x("./matrices/can_838.mtx");
    assert_eq!(
        join(x.get_data().iter().map(|x| format!("{:.3}", x)), &","),
        join(result.get_data().iter().map(|x| format!("{:.3}", x)), &",")
    );

    let (x, result) = solve_random_system(2000, (70.0, 100.0), (-2.0, 2.0), (-20.0, 20.0), 0.2);
    assert_eq!(
        join(x.get_data().iter().map(|x| format!("{:.2}", x)), &","),
        join(result.get_data().iter().map(|x| format!("{:.2}", x)), &",")
    );
}

fn solve_file_random_x(path: &str) -> (DenseVec<f64>, DenseVec<f64>) {
    let mut x = vec![];
    let path = Path::new(path);
    let mut rng = rand::thread_rng();
    let matrix = from_file(path).unwrap();
    let n = matrix.num_rows();
    let sum = matrix.elements().fold(0.0, |sum, (_, _, val)| val + sum);
    let avg = sum / matrix.num_nonzero() as f64;
    for _ in 0..n {
        x.push(rng.gen_range(avg * 0.7, avg * 1.3));
    }
    let x = DenseVec::new(x);
    let b = &matrix * &x;
    (x, gmres(&matrix, &b, 0.0000001, 100000000, 30).unwrap())
}

fn solve_random_system(
    size: usize,
    diagonal_range: (f64, f64),
    sparse_range: (f64, f64),
    x_range: (f64, f64),
    density: f64,
) -> (DenseVec<f64>, DenseVec<f64>) {
    let mut x = vec![];
    let mut rng = rand::thread_rng();
    let mut matrix: Vec<MatrixElement<f64>> = vec![];
    for i in 0..size {
        x.push(rng.gen_range(x_range.0, x_range.1));
        for j in 0..size {
            if i == j {
                let diag_val = rng.gen_range(diagonal_range.0, diagonal_range.1);
                if rng.gen() {
                    matrix.push(MatrixElement(i, j, diag_val));
                } else {
                    matrix.push(MatrixElement(i, j, -diag_val));
                }
            } else if rng.gen_range(0.0, 1.0) < density {
                let sparse_val = rng.gen_range(sparse_range.0, sparse_range.1);
                matrix.push(MatrixElement(i, j, sparse_val));
            }
        }
    }
    let matrix = SparseMatrix::new(size, size, matrix).unwrap();
    let x = DenseVec::new(x);

    let b = &matrix * &x;
    (x, gmres(&matrix, &b, 0.00000001, 100000000, 30).unwrap())
}
