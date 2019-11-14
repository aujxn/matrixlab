/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

mod matrix {
    use matrixlab::error::Error;
    use matrixlab::matrix::sparse::SparseMatrix;
    use matrixlab::MatrixElement;

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
        assert_eq!(mat.num_columns(), 2);
        assert_eq!(mat.num_rows(), 2);
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

    #[test]
    fn number_of_nonzeros_and_get_data() {
        let elements = vec![
            MatrixElement(0, 0, 2u64),
            MatrixElement(0, 1, 1),
            MatrixElement(1, 0, 3),
            MatrixElement(1, 1, 7),
            MatrixElement(3, 2, 11),
        ];
        // Create a new 4X4 matrix
        let mat = SparseMatrix::new(4, 4, elements.clone()).unwrap();
        let data = [2u64, 1, 3, 7, 11];
        let rows = [0usize, 2, 4, 4, 5];
        let cols = [0usize, 1, 0, 1, 2];

        assert_eq!(mat.num_nonzero(), 5);
        assert_eq!(mat.get_data(), &data);
        assert_eq!(mat.get_rows(), &rows);
        assert_eq!(mat.get_columns(), &cols);
    }

    #[test]
    #[should_panic]
    fn get_row() {
        let elements = vec![
            MatrixElement(0, 0, 2u64),
            MatrixElement(0, 1, 1),
            MatrixElement(1, 0, 3),
            MatrixElement(1, 1, 7),
            MatrixElement(3, 2, 11),
        ];
        // Create a new 4X4 matrix
        let mat = SparseMatrix::new(4, 4, elements.clone()).unwrap();
        let data = [2u64, 1, 3, 7, 11];
        let cols = [0usize, 1, 0, 1, 2];

        assert_eq!(mat.get_row(0), (&data[0..2], &cols[0..2]));
        assert_eq!(mat.get_row(1), (&data[2..4], &cols[2..4]));
        assert_eq!(mat.get_row(2), (&data[4..4], &cols[4..4]));
        assert_eq!(mat.get_row(3), (&data[4..5], &cols[4..5]));
        mat.get_row(5);
    }

    #[test]
    fn get_row_sums() {
        let elements = vec![
            MatrixElement(0, 0, 2u64),
            MatrixElement(0, 1, 1),
            MatrixElement(1, 0, 3),
            MatrixElement(1, 1, 7),
            MatrixElement(3, 2, 11),
        ];
        // Create a new 4X4 matrix
        let mat = SparseMatrix::new(4, 4, elements.clone()).unwrap();
        let row_sums = mat.row_sums();

        assert_eq!(row_sums, vec![3, 10, 0, 11]);
    }
}

mod transpose {
    use matrixlab::matrix::sparse::SparseMatrix;
    use matrixlab::MatrixElement;

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
    use matrixlab::matrix::sparse::SparseMatrix;
    use matrixlab::vector::dense::DenseVec;
    use matrixlab::MatrixElement;

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
        // Create a new 3X3 matrix
        let mat = SparseMatrix::new(3, 3, elements.clone()).unwrap();
        let result = &mat * &DenseVec::new(vec![7, 2, 1]);

        //Check to make sure we got the same elements back
        assert_eq!(result, DenseVec::new(vec![16, 35, 11]));
    }
}

mod sparse_vector_mult {
    use matrixlab::matrix::sparse::SparseMatrix;
    use matrixlab::vector::sparse::SparseVec;
    use matrixlab::MatrixElement;

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
        assert_eq!(
            result,
            SparseVec::new(vec![(0, 16), (1, 35), (2, 11)], 3).unwrap()
        );
    }
}

mod sparse_matrix_mult {
    use matrixlab::matrix::sparse::SparseMatrix;
    use matrixlab::MatrixElement;

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
    use matrixlab::matrix::sparse::SparseMatrix;
    use matrixlab::MatrixElement;

    #[test]
    fn all_elements_iter() {
        let elements = vec![MatrixElement(0, 0, 2i64), MatrixElement(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();
        let all_elements = mat
            .all_elements()
            .map(|(_, _, val)| match val {
                Some(val) => *val,
                None => 0,
            })
            .collect::<Vec<i64>>();
        assert_eq!(vec![2i64, 1, 0, 0], all_elements);
    }

    #[test]
    fn row_iter() {
        let elements = vec![
            MatrixElement(0, 0, 2u64),
            MatrixElement(0, 1, 1),
            MatrixElement(1, 0, 3),
            MatrixElement(1, 1, 7),
            MatrixElement(3, 2, 11),
        ];
        // Create a new 3X3 matrix
        let mat = SparseMatrix::new(4, 4, elements.clone()).unwrap();

        let mut row_iter = mat.row_iter();
        let data = [2u64, 1, 3, 7, 11];
        let cols = [0usize, 1, 0, 1, 2];

        assert_eq!(row_iter.next(), Some((&cols[0..2], &data[0..2])));
        assert_eq!(row_iter.next(), Some((&cols[2..4], &data[2..4])));
        assert_eq!(row_iter.next(), Some((&cols[4..4], &data[4..4])));
        assert_eq!(row_iter.next(), Some((&cols[4..5], &data[4..5])));
        assert_eq!(row_iter.next(), None);
    }

}
