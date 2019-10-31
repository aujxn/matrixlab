/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

mod matrix {
    use crate::matrix::dense::DenseMatrix;
    #[test]
    fn matrix_multiplication() {
        let columns = vec![vec![1, 0], vec![2, 1]];
        // Create a new 2X2 matrix
        let mat = DenseMatrix::new(columns);

        let columns = vec![vec![1, 0], vec![0, 1]];
        // Create a new 2X2 matrix
        let other_mat = DenseMatrix::new(columns);

        //Check to make sure we got the same elements back
        assert_eq!(mat, mat.safe_mul(&other_mat).unwrap());
    }
    #[test]
    fn vector_multiplication() {
        let columns = vec![vec![1, 0], vec![2, 1]];
        // Create a new 2X2 matrix
        let mat = DenseMatrix::new(columns);

        //Check to make sure we got the same elements back
        assert_eq!(vec![3, 1], mat.vec_mul(&vec![1, 1]).unwrap());
    }
    #[test]
    fn backsolve() {
        let columns = vec![vec![1.0, 0.0], vec![2.0, 1.0]];
        // Create a new 2X2 matrix
        let mat = DenseMatrix::new(columns);

        //Check to make sure we got the same elements back
        assert_eq!(vec![1.0, 2.0], mat.backsolve(&vec![5.0, 2.0]));
    }
    #[test]
    fn least_squares_simple() {
        let columns = vec![vec![1.0, 0.0], vec![2.0, 1.0]];
        // Create a new 2X2 matrix
        let mat = DenseMatrix::new(columns);

        //Check to make sure we got the same elements back
        assert_eq!(vec![1.0, 2.0], mat.least_squares(&vec![5.0, 2.0]).unwrap());
    }
    //#[test]
    //fn least_squares() {
    //    let columns = vec![vec![1.0,1.0],vec![2.0,1.0]];
    //    // Create a new 2X2 matrix
    //    let mat = DenseMatrix::new(columns);

    //    //Check to make sure we got the same elements back
    //    assert_eq!(vec![1.0,2.0],mat.least_squares(&vec![5.0,3.0]));
    //}
    #[test]
    fn scalar_multiplication() {
        let columns = vec![vec![1, 0], vec![2, 1]];
        // Create a new 2X2 matrix
        let mat = DenseMatrix::new(columns);

        //This matrix contains every value from our
        //first matrix, doubled. This is what we
        //should get when we multiply the matrix by two.
        let columns = vec![vec![2, 0], vec![4, 2]];
        // Create a new 2X2 matrix
        let other_mat = DenseMatrix::new(columns);

        //Check to make sure we got the same elements back
        assert_eq!(other_mat, mat.scale(&2));
    }
}
mod orthogonal {
    use crate::matrix::dense::DenseMatrix;
    #[test]
    fn matrix_multiplication() {
        let columns = vec![
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        // Create a new 3X3 matrix
        let mat: DenseMatrix<f64> = DenseMatrix::new(columns);

        let columns = vec![
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ];
        // Create the expected 3X3 matrix
        let other_mat = DenseMatrix::new(columns);

        //Check to make sure we got the same elements back
        assert_eq!(other_mat, mat.factor_q());
    }
}
