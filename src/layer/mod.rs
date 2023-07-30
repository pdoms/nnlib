pub mod dense;
use crate::la::Matrix;

pub trait Layer
{
    fn forward(&mut self, inputs: &Matrix<f64>);
    fn backward(&mut self, dvals: &Matrix<f64>);
}
