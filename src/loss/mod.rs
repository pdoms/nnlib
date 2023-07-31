//This code is heavily derived from www.nnfs.io and all credit goes to
//Harrison Kinsley & Daniel Kukieła
//
//That being said this very code, the Rust version, is under MIT license. 


use crate::{la::Matrix, activation::{Softmax, Activation}};

pub fn calculate_loss<L>(loss: L, y_pred: &Matrix<f64>, y_true: &Matrix<i32>) -> f64
where
    L: Loss,
{
    let sample_losses = loss.forward(y_pred, y_true);
    sample_losses.mean()
}

pub trait Loss 
{
    fn forward(&self, y_pred: &Matrix<f64>, y_true: &Matrix<i32>) -> Matrix<f64>;
    fn backward(&mut self, dvalues: Matrix<f64>, y_true: &Matrix<i32>);
}

const CCEMINMAXCLIP: f64 = 1e-7; 
pub struct CatCrossEnt {
    dinputs: Matrix<f64>
}

pub struct SoftmaxCatCrossEnt {
    pub activation: Softmax,
    pub output: Matrix<f64>,
    pub dinputs: Matrix<f64>
}

impl CatCrossEnt {
    pub fn new() -> Self {
        Self { dinputs: Matrix::new((1,1)) }
    }
}

impl SoftmaxCatCrossEnt {
    pub fn new() -> Self {
        Self { 
            activation: Softmax::new(), 
            output: Matrix::new((1,1)), 
            dinputs: Matrix::new((1,1)) }
    }

    pub fn forward(&mut self, inputs: &Matrix<f64>, y_true: &Matrix<i32>) -> f64 {
        self.activation.forward(&inputs);
        self.output = self.activation.output.clone();
        calculate_loss(CatCrossEnt::new(), &self.output, y_true)
    }

    pub fn backward(&mut self, dvalues: &Matrix<f64>, y_true: &Matrix<i32>) {
        let samples = dvalues.rows;
        let mut y = y_true.clone();
        if !y_true.is_vector() {
            y = y_true.argmax_axis(1);
        }
        self.dinputs = dvalues.clone();
        for i in 0..samples {
            let idx = y.get((0, i));
            let v = self.dinputs.get((i, idx as usize));
            self.dinputs.set(v - 1.0, (i, idx as usize));
        }
        let scalar = Matrix::like_with(&self.dinputs, samples as f64);
        self.dinputs = self.dinputs.clone() / scalar    
    }
}

impl Loss for CatCrossEnt {
    fn forward(&self, y_pred: &Matrix<f64>, y_true: &Matrix<i32>) -> Matrix<f64> {
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

    fn backward(&mut self, dvalues: Matrix<f64>, y_true: &Matrix<i32>) {
        let samples = dvalues.rows;
        let labels = dvalues.cols;

        if y_true.is_vector() {
            let y_true_one_hot = Matrix::<i32>::zeroed((labels, labels));   
            for i in 0..y_true.len {
                let idx = y_true.get((0, i));
                y_true_one_hot.set(1, (i, idx as usize));
            }
            let r = -Matrix::<f64>::from(y_true_one_hot);
            self.dinputs = r / dvalues;
            self.dinputs = self.dinputs.clone() / Matrix::like_with(&self.dinputs, samples as f64);
        }
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

        let loss = calculate_loss(CatCrossEnt::new(), &softmax_out.clone(), &class_targets);
        let loss_idx = calculate_loss(CatCrossEnt::new(), &softmax_out, &index_targets);
        let exp = 0.38506088005216804;
        assert_eq!(loss, exp);
        assert_eq!(loss_idx, exp);
    }

    #[test]
    fn loss_cat_cross_ent_back() {
        let softmax_out = Matrix::from_vec2(vec![vec![0.7, 0.1, 0.2],
vec![0.1, 0.5, 0.4],
vec![0.02, 0.9, 0.08]]);
        let class_targets = Matrix::from_vec(vec![0,1,1]);
        let mut loss = CatCrossEnt::new();
        loss.backward(softmax_out.clone(), &class_targets);
        let res1 = Matrix::from_vec2(
 vec![
 	vec![-0.4761904761904762, -0.0, -0.0],
 	vec![-0.0, -0.6666666666666666, -0.0],
 	vec![-0.0, -0.3703703703703704, -0.0]
 ]);
        assert_eq!(loss.dinputs, res1);
        //combined
        let mut combined = SoftmaxCatCrossEnt::new();
        combined.backward(&softmax_out, &class_targets);
     let res2 = Matrix::from_vec2(
     vec![
 	vec![-0.10000000000000002, 0.03333333333333333, 0.06666666666666667],
 	vec![0.03333333333333333, -0.16666666666666666, 0.13333333333333333],
 	vec![0.006666666666666667, -0.033333333333333326, 0.02666666666666667]
     ]);
     assert_eq!(combined.dinputs, res2);
    }
}

