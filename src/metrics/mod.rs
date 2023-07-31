use crate::la::Matrix;

pub fn softmax_accuracy(outputs: &Matrix<f64>, y: &Matrix<i32>) -> f64 {
    let predictions = outputs.argmax_axis(1);
    let mut y_true = (*y).clone();
    if !y.is_vector() {
        y_true = (*y).argmax_axis(1)
    }
    let truthiness = predictions.truthiness(&y);
    return truthiness.mean();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn acc_softmax() {
        let softmax_out = Matrix::from_vec2(vec![
            vec![0.7, 0.2, 0.1],
            vec![0.5, 0.1, 0.4],
            vec![0.02, 0.9, 0.08]
        ]);
        let class_targets = Matrix::from_vec(vec![0,1,1]);
        assert_eq!(softmax_accuracy(&softmax_out,&class_targets), 0.6666666666666666);
    }
}

