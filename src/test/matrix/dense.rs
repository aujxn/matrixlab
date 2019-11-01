/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

mod matrix {
    use crate::matrix::dense::DenseMatrix;
    use crate::vector::dense::DenseVec;
    use crate::MatrixElement;
    use crate::error::Error;
    use ndarray::Array;
    use std::iter::FromIterator;

    #[test]
    fn valid_construction_from_triplets() {
        let elements = vec![(1usize, 4usize, 12u64), (3, 3, 8), (3, 3, 10), (1, 0, 5), (2, 2, 7), (0, 4, 3)];
        let elements = elements.iter().map(|&(i, j, val)| MatrixElement(i, j, val)).collect();

        let matrix = DenseMatrix::from_triplets(4, 5, elements).unwrap();
        let result: Vec<u64> = matrix.get_data().iter().filter(|x| **x != 0).map(|&x| x).collect();

        assert_eq!(vec![3u64, 5, 12, 7, 18], result);
    }
        
    #[test]
    fn error_construction_from_triplets() {
        let error = DenseMatrix::from_triplets(3, 3, vec![MatrixElement(3, 2, 15)]);
        assert_eq!(Err(Error::ElementOutOfBounds), error);
    }

    #[test]
    fn dense_matrix_multiplication() {
        // Create a new 4X2 matrix
        let matrix = vec![12u64, 15, 7, 22, 3, 0, 8, 19];
        let mat = DenseMatrix::new(4, 2, matrix);

        // Create a new 2X2 matrix
        let matrix2 = vec![3u64, 8, 9, 13];
        let mat2 = DenseMatrix::new(2, 2, matrix2);

        // Create result
        let result = vec![171, 291, 219, 342, 9, 24, 195, 311];
        let result = DenseMatrix::new(4, 2, result);

        assert_eq!(result, mat.safe_dense_mat_mul(&mat2).unwrap());
    }
    
    #[test]
    fn dense_vector_multiplication() {
        // Create a new 2X4 matrix
        let matrix = vec![12u64, 15, 7, 22, 3, 0, 8, 19];
        let mat = DenseMatrix::new(2, 4, matrix);

        assert_eq!(DenseVec::new(vec![691, 484]), mat.dense_vec_mul(&DenseVec::new(vec![3u64, 7, 0, 25])));
    }

    #[test]
    fn dense_matrix_multiplication_and_transpose() {
        // Create a new 4X2 matrix
        let matrix = vec![12u64, 15, 7, 22, 3, 0, 8, 19];
        let mat = DenseMatrix::new(4, 2, matrix);
        let t = mat.transpose();

        let result = vec![369, 414, 36, 381, 414, 533, 21, 474, 36, 21, 9, 24, 381, 474, 24, 425];
        assert_eq!(DenseMatrix::new(4, 4, result), mat.safe_dense_mat_mul(&t).unwrap());
    }
}

    /*
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
*/
