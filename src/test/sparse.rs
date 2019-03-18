mod matrix {
    use crate::error::Error;
    use crate::matrix::sparse::{Element,Matrix};
    #[test]
    fn matrix_creation() {
        let elements = vec![Element(1,1,2u64),Element(1,2,1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone()).unwrap();

        let new_elements = mat.elements();

        //Check to make sure we got the same elements back
        assert_eq!(elements,new_elements.collect::<Vec<Element<u64>>>());
    }
    #[test]
    fn out_of_bounds_matrix_creation() {
        let elements = vec![Element(1,1,2u64),Element(1,3,1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone());


        //Check to make sure that the right error got returned
        assert_eq!(mat, Err(Error::ElementOutOfBounds));
    }
    #[test]
    fn full_matrix_creation() {
        let elements = vec![
            Element(1,1,2u64),
            Element(1,2,1),
            Element(2,1,3),
            Element(2,2,7),
            ];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone()).unwrap();

        let new_elements = mat.elements();

        //Check to make sure we got the same elements back
        assert_eq!(elements,new_elements.collect::<Vec<Element<u64>>>());
    }
    #[test]
    fn matrix_sum() {
        let elements = vec![Element(1,1,2u64),Element(1,2,1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements).unwrap();

        let new_elements = mat.elements();

        //Check to make sure we got the same elements back
        assert_eq!(3u64,new_elements.map(|Element(_,_,d)| d).sum());
    }
}


mod transpose {
    use crate::matrix::sparse::{Element,Matrix};
    #[test]
    fn transpose_sum() {
        let elements = vec![Element(1,1,2u64),Element(1,2,17)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements).unwrap();
        let new_mat = mat.transpose();

        let new_elements = new_mat.elements();

        //Check to make sure we got the same elements back
        assert_eq!(19u64,new_elements.map(|Element(_,_,d)| d).sum());
    }

}
mod vector_mult {
    use crate::matrix::sparse::{Element,Matrix};
    #[test]
    fn mult() {
        let elements = vec![Element(1,1,2u64),Element(1,2,1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone()).unwrap();

        let result: Vec<u64> = &mat * &vec![2,1];

        //Check to make sure we got the same elements back
        assert_eq!(result,vec![5,0]);
    }
    #[test]
    fn identity_mult() {
        let elements = vec![
            Element(1,1,2u64),
            Element(1,2,1),
            Element(2,1,3),
            Element(2,2,7),
            ];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone()).unwrap();
        let result: Vec<u64> = &mat * &vec![1,1];

        //Check to make sure we got the same elements back
        assert_eq!(result,vec![3,10]);
    }
}
mod sparse_vector_mult {
    use crate::matrix::sparse::{Element,Matrix};
    #[test]
    fn mult() {
        let elements = vec![Element(1,1,2u64),Element(1,2,1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone()).unwrap();

        let result: Vec<Element<u64>> = &mat * &vec![Element(1,1,1),Element(2,1,1)];

        //Check to make sure we got the same elements back
        assert_eq!(result,vec![Element(1,1,3)]);
    }
    #[test]
    fn identity_mult() {
        let elements = vec![
            Element(1,1,2u64),
            Element(1,2,1),
            Element(2,1,3),
            Element(2,2,7),
            ];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone()).unwrap();
        let result: Vec<Element<u64>> = &mat * &vec![Element(1,1,1),Element(2,1,1)];

        //Check to make sure we got the same elements back
        assert_eq!(result,vec![Element(1,1,3),Element(2,1,10)]);
    }
}
mod matrix_mult {
    use crate::matrix::sparse::{Element,Matrix};
    #[test]
    fn mult() {
        let elements = vec![Element(1,1,2i64),Element(1,2,1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone()).unwrap();
        let new_mat = mat.transpose();

        let result: Matrix<i64> = &mat * &new_mat;

        let elements = vec![Element(1,1,5i64),];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone()).unwrap();

        //Check to make sure we got the same elements back
        assert_eq!(result,mat);
    }
}
mod iter {
    use crate::matrix::sparse::{Element,Matrix};
    #[test]
    fn all_elements_iter() {
        let elements = vec![Element(1,1,2i64),Element(1,2,1)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone()).unwrap();
        let all_elements = mat.all_elements();
        assert_eq!(vec![2i64,1,0,0],all_elements.collect::<Vec<i64>>());

    }

}
