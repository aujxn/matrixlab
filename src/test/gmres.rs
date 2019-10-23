mod matrix {
    use crate::error::Error;
    use crate::matrix::sparse::{MatElement, SparseMatrix};
    #[test]
    fn iteration_test() {
        let elements = vec![MatElement(0, 0, 2f64), MatElement(1, 1, 2f64), MatElement(0, 1, 1.0)];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();

        let result = mat.gmres(vec![3.0, 2.0], 1000, 1.0 / 1000000.0, 50);
        assert!(result.is_ok());
    }
    #[test]
    fn exact_test() {
        let elements = vec![MatElement(0, 0, 2f64), MatElement(1, 1, 2f64), MatElement(0, 1, 1.0)];
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
            MatElement(0, 0, 1f64),
            MatElement(0, 1, 1f64),
            MatElement(1, 0, 0f64),
            MatElement(1, 1, 0.0),
        ];
        // Create a new 2X2 matrix
        let mat = SparseMatrix::new(2, 2, elements.clone()).unwrap();

        let result = mat.gmres(vec![1.1, 0.9], 100, 1.0 / 1000000.0, 50);
        assert_eq!(result, Err(Error::ExceededIterations(vec![])));
    }
}
