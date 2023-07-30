//This code is heavily derived from www.nnfs.io and all credit goes to
//Harrison Kinsley & Daniel Kukieła
//
//That being said this very code, the Rust version, is under MIT license. 


use crate::la::Matrix;

pub fn calculate_loss<L>(loss: L, y_pred: Matrix<f64>, y_true: Matrix<i32>) -> f64
where
    L: Loss,
{
    let sample_losses = loss.forward(y_pred, y_true);
    sample_losses.mean()
}

pub trait Loss 
{
    fn forward(&self, y_pred: Matrix<f64>, y_true: Matrix<i32>) -> Matrix<f64>;
}

const CCEMINMAXCLIP: f64 = 1e-7; 
pub struct CatCrossEnt;

impl Loss for CatCrossEnt {
    fn forward(&self, y_pred: Matrix<f64>, y_true: Matrix<i32>) -> Matrix<f64> {
        let confidences: Matrix<f64>;
        y_pred.clip(CCEMINMAXCLIP, 1.0-CCEMINMAXCLIP);
        if y_true.is_vector() {
            assert!(y_pred.rows == y_true.len, "Labels don't match: {} (y_pred.rows) != {} (y_true.len)", y_pred.rows, y_true.len);
            confidences = Matrix::new((0, y_pred.rows));
            for i in 0..y_true.len {
                let idx = y_true.get((0, i)) as usize;
                let v = y_pred.get((i, idx));
                confidences.set(v, (0,i));
            }
        } else {
            let r = y_pred * Matrix::<f64>::from(y_true);
            confidences = r.sum_axis(1, false);
        }
        confidences.neg_ln();
        confidences
    }
}



#[cfg(test)]
mod test {
    use crate::la::Matrix;

    use super::*;

    #[test]
    fn loss_cat_cross_ent() {
        let softmax_out = Matrix::from_vec2(vec![
                vec![0.7,  0.1, 0.2],
                vec![0.1,  0.5, 0.4], 
                vec![0.02, 0.9, 0.08]
        ]);
        let class_targets = Matrix::from_vec2(vec![
                vec![1, 0, 0],
                vec![0, 1, 0],
                vec![0, 1, 0]
        ]);

        let index_targets = Matrix::from_vec(vec![0, 1, 1]);

        let loss = calculate_loss(CatCrossEnt, softmax_out.clone(), class_targets);
        let loss_idx = calculate_loss(CatCrossEnt, softmax_out, index_targets);
        let exp = 0.38506088005216804;
        assert_eq!(loss, exp);
        assert_eq!(loss_idx, exp);
    }
}

