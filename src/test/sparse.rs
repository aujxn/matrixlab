mod matrix {
    use crate::error::Error;
    use crate::matrix::sparse::{Element, Matrix};
    #[test]
    fn matrix_creation() {
        let elements = vec![Element(0, 0, 2u64), Element(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements.clone()).unwrap();

        let new_elements: Vec<Element<u64>> = mat
            .elements()
            .map(|(i, j, val)| Element(i, j, *val))
            .collect();

        //Check to make sure we got the same elements back
        assert_eq!(elements, new_elements);
    }

    #[test]
    fn out_of_bounds_matrix_creation() {
        let elements = vec![Element(1, 1, 2u64), Element(1, 3, 1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements.clone());

        //Check to make sure that the right error got returned
        assert_eq!(mat, Err(Error::ElementOutOfBounds));
    }

    #[test]
    fn full_matrix_creation() {
        let elements = vec![
            Element(0, 0, 2u64),
            Element(0, 1, 1),
            Element(1, 0, 3),
            Element(1, 1, 7),
        ];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements.clone()).unwrap();

        let new_elements: Vec<Element<u64>> = mat
            .elements()
            .map(|(i, j, val)| Element(i, j, *val))
            .collect();

        //Check to make sure we got the same elements back
        assert_eq!(elements, new_elements);
    }

    #[test]
    fn matrix_sum() {
        let elements = vec![Element(0, 0, 2u64), Element(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements).unwrap();

        let sum: u64 = mat.elements().fold(0, |acc, (_, _, val)| acc + *val);

        //Check to make sure we got the same elements back
        assert_eq!(3u64, sum);
    }
}

mod transpose {
    use crate::matrix::sparse::{Element, Matrix};
    #[test]
    fn transpose() {
        let elements = vec![Element(0, 0, 2u64), Element(0, 1, 17)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements).unwrap();
        let new_mat = mat.transpose();

        let new_elements: Vec<(usize, usize, &u64)> = new_mat.elements().collect();

        //Check to make sure we got the same elements back
        assert_eq!(vec![(0, 0, &2u64), (1, 0, &17u64)], new_elements);
    }
}

mod vector_mult {
    use crate::matrix::sparse::{Element, Matrix};
    #[test]
    fn mult() {
        let elements = vec![Element(0, 0, 2u64), Element(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements.clone()).unwrap();

        let result: Vec<u64> = &mat * &vec![2, 1];

        //Check to make sure we got the same elements back
        assert_eq!(result, vec![5, 0]);
    }

    #[test]
    fn identity_mult() {
        let elements = vec![
            Element(0, 0, 2u64),
            Element(0, 1, 1),
            Element(1, 0, 3),
            Element(1, 1, 7),
        ];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements.clone()).unwrap();
        let result: Vec<u64> = &mat * &vec![1, 1];

        //Check to make sure we got the same elements back
        assert_eq!(result, vec![3, 10]);
    }

    #[test]
    fn bigger_mult() {
        let elements = vec![
            Element(0, 0, 2u64),
            Element(0, 1, 1),
            Element(1, 0, 3),
            Element(1, 1, 7),
            Element(2, 2, 11),
        ];
        // Create a new 2X2 matrix
        let mat = Matrix::new(3, 3, elements.clone()).unwrap();
        let result: Vec<u64> = &mat * &vec![7, 2, 1];

        //Check to make sure we got the same elements back
        assert_eq!(result, vec![16, 35, 11]);
    }
}

mod sparse_vector_mult {
    use crate::matrix::sparse::{Element, Matrix};
    #[test]
    fn mult() {
        let elements = vec![Element(0, 0, 2u64), Element(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements.clone()).unwrap();

        let result: Vec<Element<u64>> = &mat * &vec![Element(0, 0, 1), Element(1, 0, 1)];

        //Check to make sure we got the same elements back
        assert_eq!(result, vec![Element(0, 0, 3)]);
    }

    #[test]
    fn identity_mult() {
        let elements = vec![
            Element(0, 0, 2u64),
            Element(0, 1, 1),
            Element(1, 0, 3),
            Element(1, 1, 7),
        ];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements.clone()).unwrap();
        let result: Vec<Element<u64>> = &mat * &vec![Element(0, 0, 1), Element(1, 0, 1)];

        //Check to make sure we got the same elements back
        assert_eq!(result, vec![Element(0, 0, 3), Element(1, 0, 10)]);
    }

    #[test]
    fn bigger_mult() {
        let elements = vec![
            Element(0, 0, 2u64),
            Element(0, 1, 1),
            Element(1, 0, 3),
            Element(1, 1, 7),
            Element(2, 2, 11),
        ];
        // Create a new 2X2 matrix
        let mat = Matrix::new(3, 3, elements.clone()).unwrap();
        let result: Vec<Element<u64>> =
            &mat * &vec![Element(0, 0, 7), Element(1, 0, 2), Element(2, 0, 1)];

        //Check to make sure we got the same elements back
        assert_eq!(
            result,
            vec![Element(0, 0, 16), Element(1, 0, 35), Element(2, 0, 11)]
        );
    }
}

mod matrix_mult {
    use crate::matrix::sparse::{Element, Matrix};
    #[test]
    fn mult() {
        let elements = vec![Element(0, 0, 2i64), Element(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements.clone()).unwrap();
        let new_mat = mat.transpose();

        let result: Matrix<i64> = &mat * &new_mat;

        let elements = vec![Element(0, 0, 5i64)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements.clone()).unwrap();

        //Check to make sure we got the same elements back
        assert_eq!(result, mat);
    }
}

mod iter {
    use crate::matrix::sparse::{Element, Matrix};
    #[test]
    fn all_elements_iter() {
        let elements = vec![Element(0, 0, 2i64), Element(0, 1, 1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2, 2, elements.clone()).unwrap();
        let all_elements = mat
            .all_elements()
            .map(|(_, _, val)| *val)
            .collect::<Vec<i64>>();
        print!("{:?}", all_elements);
        assert_eq!(vec![2i64, 1, 0, 0], all_elements);
    }

}
