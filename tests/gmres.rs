/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

mod matrix {
    use matrixlab::matrix::sparse::SparseMatrix;
    use matrixlab::MatrixElement;
    use matrixlab::vector::dense::DenseVec;
    use matrixlab::matrix::gmres::gmres;

    #[test]
    fn five_by_five_gmres() {
        let A = [(0usize, 0usize, 12.0f64), (0, 2, 9.0), (0, 3, 4.0), (1, 1, 18.0),
            (2, 2, 6.0), (3, 4, 9.0), (4, 0, 1.0)].iter().map(|x| MatrixElement::new(x.0, x.1, x.2)).collect();
        let A = SparseMatrix::new(5, 5, A).unwrap();
        let x = DenseVec::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);

        let b = &A * &x;

        let mut result = gmres(A, b, 1.0 / 10000.0, 100000, 5).unwrap();

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
