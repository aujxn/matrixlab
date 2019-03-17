mod matrix {
    use crate::matrix::sparse::{Element,Matrix};
    #[test]
    fn iteration_test() {
        let elements = vec![
            Element(1,1,2f64),
            Element(2,2,2f64),
            Element(1,2,1.0)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone()).unwrap();

        let result = mat.gmres(vec![3.0,2.0]);
        assert!(result.is_ok());
    }
    #[test]
    fn close_test() {
        let elements = vec![
            Element(1,1,2f64),
            Element(2,2,2f64),
            Element(1,2,1.0)];
        // Create a new 2X2 matrix
        let mat = Matrix::new(2,2,elements.clone()).unwrap();

        let result = mat.gmres(vec![3.0,2.0]).unwrap();
        assert_eq!(result,vec![1.0,1.0]);
    }
}
