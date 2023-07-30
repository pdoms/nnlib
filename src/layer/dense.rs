//This code is heavily derived from www.nnfs.io and all credit goes to
//Harrison Kinsley & Daniel Kukieła
//
//That being said this very code, the Rust version, is under MIT license. 

use crate::la::Matrix;
use crate::layer::Layer;

pub struct Dense 
{
    pub weights: Matrix<f64>,
    pub dweights: Matrix<f64>,
    pub biases: Matrix<f64>,
    pub dbiases: Matrix<f64>,
    pub inputs: Matrix<f64>,
    pub dinputs: Matrix<f64>,
    pub outputs: Matrix<f64>,
}

impl Layer for Dense {
    fn forward(&mut self, inputs: &Matrix<f64>) {
        self.inputs = inputs.clone();
        let dot = self.inputs.dot(&self.weights) + self.biases.clone();
        self.outputs = dot.clone()
    }

    fn backward(&mut self, dvals: &Matrix<f64>) {
        self.inputs.transpose();
        self.dweights = self.inputs.dot(dvals);
        self.dbiases = dvals.sum_axis(0, true);
        self.dweights.transpose();
        self.dinputs = dvals.dot(&self.dweights);
    }
}

impl Dense {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        Self {
            weights: Matrix::random((n_inputs, n_neurons), -1.0..1.0),
            dweights:Matrix::new((0,1)),
            biases: Matrix::zeroed((0, n_neurons)),
            dbiases: Matrix::new((0,1)),
            inputs: Matrix::new((0,1)),
            dinputs: Matrix::new((0,1)),
            outputs: Matrix::new((0,1)),
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    
    #[test]
    fn dense_forward() {
        
        let data = Matrix::from_vec2(vec![vec![ 0.        ,  0.        ],
             vec![ 0.11061079,  0.01053243],
             vec![ 0.22218466, -0.00408555],
             vec![ 0.26955076, -0.19609564],
        ]);
        
        let mut layer = Dense::new(2, 3);
        layer.forward(&data);
        assert_eq!(layer.outputs.shape(), (4,3))
    }
}
