/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

mod huge {
    use itertools::join;
    use matrixlab::matrix::gmres::gmres;
    use matrixlab::matrix::sparse::SparseMatrix;
    use matrixlab::vector::dense::DenseVec;
    use matrixlab::MatrixElement;
    use rand::prelude::*;

    #[test]
    fn very_large_test_with_random() {
        let mut matrix: Vec<MatrixElement<f64>> = vec![];
        let mut x = vec![];
        let n = 200;

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
                } else if rng.gen_range(0, 100) > 80 {
                    matrix.push(MatrixElement(i, j, rng.gen_range(-10.0, 10.0)));
                }
            }
        }

        let matrix = SparseMatrix::new(n, n, matrix).unwrap();
        let x = DenseVec::new(x);

        let b = &matrix * &x;

        let result = gmres(matrix.clone(), b.clone(), 0.000001, 1000000, 50).unwrap();
        assert_eq!(
            join(x.get_data().iter().map(|x| format!("{:.2}", x)), &","),
            join(result.get_data().iter().map(|x| format!("{:.2}", x)), &",")
        );
    }
}
mod matrix {
    use matrixlab::matrix::gmres::gmres;
    use matrixlab::matrix::sparse::SparseMatrix;
    use matrixlab::vector::dense::DenseVec;
    use matrixlab::MatrixElement;

    #[test]
    fn five_by_five_gmres() {
        let data = [
            (0usize, 0usize, 12.0f64),
            (0, 2, 9.0),
            (0, 3, 4.0),
            (1, 1, 18.0),
            (2, 0, 6.0),
            (2, 2, 6.0),
            (2, 4, 3.0),
            (3, 1, 2.0),
            (3, 4, 9.0),
            (4, 0, 1.0),
            (4, 2, 5.0),
        ];

        let elements = data
            .iter()
            .map(|x| MatrixElement::new(x.0, x.1, x.2))
            .collect();

        let matrix = SparseMatrix::new(5, 5, elements).unwrap();

        let x = DenseVec::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);

        let b = &matrix * &x;

        let mut result = gmres(matrix, b, 1.0 / 10000.0, 100000, 5).unwrap();

        for x in result.get_data_mut() {
            *x = x.round()
        }

        assert_eq!(x, result);
    }
}

/* gmres is currently out of order
mod matrix {
    use crate::error::Error;
    use crate::matrix::sparse::SparseMatrix; use crate::MatrixElement;
    #[test]
    fn iteration_test() {
        let elements = vec![MatrixElement(0, 0, 2f64), MatrixElement(1, 1, 2f64), MatrixElement(0, 1, 1.0)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();

        let result = mat.gmres(vec![3.0, 2.0], 1000, 1.0 / 1000000.0, 50);
        assert!(result.is_ok());
    }
    #[test]
    fn exact_test() {
        let elements = vec![MatrixElement(0, 0, 2f64), MatrixElement(1, 1, 2f64), MatrixElement(0, 1, 1.0)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();

        let result = mat
            .gmres(vec![3.0, 2.0], 100000, 1.0 / 1000000.0, 50)
            .unwrap();
        assert_eq!(result, vec![1.0, 1.0]);
    }
    #[test]
    fn failure_test() {
        let elements = vec![
            MatrixElement(0, 0, 1f64),
            MatrixElement(0, 1, 1f64),
            MatrixElement(1, 0, 0f64),
            MatrixElement(1, 1, 0.0),
        ];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();

        let result = mat.gmres(vec![1.1, 0.9], 100, 1.0 / 1000000.0, 50);
        assert_eq!(result, Err(Error::ExceededIterations(vec![])));
    }
}
*/
