/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

mod matrix {
    use crate::error::Error;
    use crate::matrix::sparse::SparseMatrix;
    use crate::MatrixElement;

    #[test]
    fn matrix_creation() {
        let elements = vec![MatrixElement(0, 0, 2u64), MatrixElement(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();

        let new_elements: Vec<MatrixElement<u64>> = mat
            .elements()
            .map(|(i, j, val)| MatrixElement(i, j, *val))
            .collect();

        //Check to make sure we got the same elements back
        assert_eq!(elements, new_elements);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_matrix_creation() {
        let elements = vec![MatrixElement(1, 1, 2u64), MatrixElement(1, 3, 1)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();
    }

    #[test]
    fn full_matrix_creation() {
        let elements = vec![
            MatrixElement(0, 0, 2u64),
            MatrixElement(0, 1, 1),
            MatrixElement(1, 0, 3),
            MatrixElement(1, 1, 7),
        ];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();

        let new_elements: Vec<MatrixElement<u64>> = mat
            .elements()
            .map(|(i, j, val)| MatrixElement(i, j, *val))
            .collect();

        //Check to make sure we got the same elements back
        assert_eq!(elements, new_elements);
    }

    #[test]
    fn matrix_sum() {
        let elements = vec![MatrixElement(0, 0, 2u64), MatrixElement(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements).unwrap();

        let sum: u64 = mat.elements().fold(0, |acc, (_, _, val)| acc + *val);

        //Check to make sure we got the same elements back
        assert_eq!(3u64, sum);
    }
}

mod transpose {
    use crate::matrix::sparse::SparseMatrix;
    use crate::MatrixElement;

    #[test]
    fn transpose() {
        let elements = vec![MatrixElement(0, 0, 2u64), MatrixElement(0, 1, 17)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements).unwrap();
        let new_mat = mat.transpose();

        let new_elements: Vec<(usize, usize, &u64)> = new_mat.elements().collect();

        //Check to make sure we got the same elements back
        assert_eq!(vec![(0, 0, &2u64), (1, 0, &17u64)], new_elements);
    }
}

mod dense_vector_mult {
    use crate::matrix::sparse::SparseMatrix;
    use crate::vector::dense::DenseVec;
    use crate::MatrixElement;

    #[test]
    fn dense_vec_mult() {
        let elements = vec![MatrixElement(0, 0, 2u64), MatrixElement(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();

        let result = &mat * &DenseVec::new(vec![2, 1]);

        //Check to make sure we got the same elements back
        assert_eq!(result, DenseVec::new(vec![5, 0]));
    }

    #[test]
    fn identity_mult() {
        let elements = vec![
            MatrixElement(0, 0, 2u64),
            MatrixElement(0, 1, 1),
            MatrixElement(1, 0, 3),
            MatrixElement(1, 1, 7),
        ];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();
        let result = &mat * &DenseVec::new(vec![1, 1]);

        //Check to make sure we got the same elements back
        assert_eq!(result, DenseVec::new(vec![3, 10]));
    }

    #[test]
    fn bigger_mult() {
        let elements = vec![
            MatrixElement(0, 0, 2u64),
            MatrixElement(0, 1, 1),
            MatrixElement(1, 0, 3),
            MatrixElement(1, 1, 7),
            MatrixElement(2, 2, 11),
        ];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(3, 3, elements.clone()).unwrap();
        let result = &mat * &DenseVec::new(vec![7, 2, 1]);

        //Check to make sure we got the same elements back
        assert_eq!(result, DenseVec::new(vec![16, 35, 11]));
    }
}

mod sparse_vector_mult {
    use crate::matrix::sparse::SparseMatrix;
    use crate::MatrixElement;
    use crate::vector::sparse::SparseVec;

    #[test]
    fn mult() {
        let elements = vec![MatrixElement(0, 0, 2u64), MatrixElement(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();

        let result = &mat * &SparseVec::new(vec![(0, 1u64), (1, 1)], 2).unwrap();

        //Check to make sure we got the same elements back
        assert_eq!(result, SparseVec::new(vec![(0, 3u64)], 2).unwrap());
    }

    #[test]
    fn identity_mult() {
        let elements = vec![
            MatrixElement(0, 0, 2u64),
            MatrixElement(0, 1, 1),
            MatrixElement(1, 0, 3),
            MatrixElement(1, 1, 7),
        ];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();
        let result = &mat * &SparseVec::new(vec![(0, 1), (1, 1)], 2).unwrap();

        //Check to make sure we got the same elements back
        assert_eq!(result, SparseVec::new(vec![(0, 3), (1, 10)], 2).unwrap());
    }

    #[test]
    fn bigger_mult() {
        let elements = vec![
            MatrixElement(0, 0, 2u64),
            MatrixElement(0, 1, 1),
            MatrixElement(1, 0, 3),
            MatrixElement(1, 1, 7),
            MatrixElement(2, 2, 11),
        ];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(3, 3, elements.clone()).unwrap();
        let result = &mat * &SparseVec::new(vec![(0, 7), (1, 2), (2, 1)], 3).unwrap();

        //Check to make sure we got the same elements back
        assert_eq!(result, SparseVec::new(vec![(0, 16), (1, 35), (2, 11)], 3).unwrap());
    }
}

mod sparse_matrix_mult {
    use crate::matrix::sparse::SparseMatrix;
    use crate::MatrixElement;

    #[test]
    fn sparse_transpose_and_mult() {
        let elements = vec![MatrixElement(0, 0, 2i64), MatrixElement(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();
        let new_mat = mat.transpose();

        let result: SparseMatrix<i64> = &mat * &new_mat;

        let elements = vec![MatrixElement(0, 0, 5i64)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();

        //Check to make sure we got the same elements back
        assert_eq!(result, mat);
    }
}

mod iter {
    use crate::matrix::sparse::SparseMatrix;
    use crate::MatrixElement;

    #[test]
    fn all_elements_iter() {
        let elements = vec![MatrixElement(0, 0, 2i64), MatrixElement(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();
        let all_elements = mat
            .all_elements()
            .map(|(_, _, val)| {
                match val {
                    Some(val) => *val,
                    None => 0,
                }
            })
            .collect::<Vec<i64>>();
        assert_eq!(vec![2i64, 1, 0, 0], all_elements);
    }
}
