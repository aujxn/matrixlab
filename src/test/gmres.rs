/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

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
