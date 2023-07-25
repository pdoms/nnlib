//This code is heavily derived from www.nnfs.io and all credit goes to
//Harrison Kinsley & Daniel Kukieła
//
//That being said this very code, the Rust version, is under MIT license. 


use crate::la::Matrix;

pub fn calculate_loss<L>(loss: L, y_pred: Matrix, y_true: Matrix) -> f64
where
    L: Loss,
{
    let sample_losses = loss.forward(y_pred, y_true);
    sample_losses.mean()
}

pub trait Loss 
{
    fn forward(&self, y_pred: Matrix, y_true: Matrix) -> Matrix;
}

const CCEMINMAXCLIP: f64 = 1e-7; 
pub struct CatCrossEnt;

impl Loss for CatCrossEnt {
    fn forward(&self, y_pred: Matrix, y_true: Matrix) -> Matrix {
        let confidences: Matrix;
        y_pred.clip(CCEMINMAXCLIP, 1.0-CCEMINMAXCLIP);
        if y_true.is_vector() {
            unimplemented!("categorical labels loss")
        } else {
            let r = y_pred * y_true;
            confidences = r.sum_axis(1, false);
        }
        confidences.neg_log()
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
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 1.0, 0.0]
        ]);

        let index_targets = Matrix::from_vec(vec![0.0, 1.0, 1.0]);

        let loss = calculate_loss(CatCrossEnt, softmax_out.clone(), class_targets);
        let loss_idx = calculate_loss(CatCrossEnt, softmax_out, index_targets);
        let exp = 0.38506088005216804;
        assert_eq!(loss, exp);
        assert_eq!(loss_idx, exp);
    }
}

