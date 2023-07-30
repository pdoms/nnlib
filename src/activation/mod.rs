//This code is heavily derived from www.nnfs.io and all credit goes to
//Harrison Kinsley & Daniel Kukieła
//
//That being said this very code, the Rust version, is under MIT license. 





use std::f64::consts::E;

use crate::la::Matrix;

pub trait Activation 
{
    fn new() -> Self;
    fn forward(&mut self, inputs: &Matrix<f64>);
    fn backward(&mut self, dvalues: Matrix<f64>);
}

const LEAK: f64 = 0.01;




pub struct ReLu
{
    output: Matrix<f64>,
    dinputs: Matrix<f64>
}

pub struct LeakyReLu
{
    output: Matrix<f64>,
    dinputs: Matrix<f64>
}

pub struct Sigmoid 
{
    output: Matrix<f64>,
    dinputs: Matrix<f64>
}

pub struct Softmax
{
    output: Matrix<f64>,
    dinputs: Matrix<f64>
}

impl Activation for ReLu {
    fn new() -> Self {
        Self {
            output: Matrix::new((1, 1)),
            dinputs: Matrix::new((1, 1))
        }
    }
    fn forward(&mut self, inputs: &Matrix<f64>) {
        self.output = inputs.map(|x| {if x >= 0.0 {x} else {0.0}});
    }

    fn backward(&mut self, dvalues: Matrix<f64>) {
        self.dinputs = dvalues.map(|x| {
            if x <= 0.0 {
                0.0
            } else {
                x
            }
        });
    }

}


impl Activation for LeakyReLu {
    fn new() -> Self {
        Self {
            output: Matrix::new((1, 1)),
            dinputs: Matrix::new((1, 1))

        }
    }
    fn forward(&mut self, inputs: &Matrix<f64>) {
        self.output = inputs.map(|x| {if x >= 0.0 {x} else {LEAK*x}})    
    }

    fn backward(&mut self, dvalues: Matrix<f64>) {
        self.dinputs = dvalues.map(|x| {if x < 0.0 {LEAK} else {1.0}})
    }
}

impl Activation for Sigmoid {
    fn new() -> Self {
        Self {
            output: Matrix::new((1, 1)),
            dinputs: Matrix::new((1, 1)),
        }
    }
    fn forward(&mut self, inputs: &Matrix<f64>) {
        self.output = inputs.map(|x| {
            1.0 / (1.0 + E.powf(-x))
        })    
    }

    fn backward(&mut self, dvalues: Matrix<f64>) {
        let f = sigmoid(&dvalues);
        self.dinputs = f.clone() * (Matrix::like_with(&f, 1.0) - f)
    }
}


pub fn sigmoid(values: &Matrix<f64>) -> Matrix<f64> {
    values.map(|x| {
            1.0 / (1.0 + E.powf(-x))
    })
}


impl Activation for Softmax {
    fn new() -> Self {
        Self {
            output: Matrix::new((1, 1)),
            dinputs: Matrix::new((1,1)),
        }
    }
    fn forward(&mut self, inputs: &Matrix<f64>) {
        let exp = (inputs.clone() - inputs.max_axis(1, true)).exp_copy();
        self.output = exp.clone() / exp.sum_axis(1, true);
    }

    fn backward(&mut self, dvalues: Matrix<f64>) {
        self.dinputs = Matrix::like(&dvalues);
    }
}


#[cfg(test)]
mod test {
    use crate::la::Matrix;
    use crate::layer::{Layer, dense::Dense};
    use super::*;


    #[test]
    fn activation_relu() {
        let data = Matrix::from_vec2(vec![
	        vec![0.0,0.0,0.0,],
	        vec![0.0038859788,-0.09141435,0.027655335,],
	        vec![-0.005156948,-0.20069951,0.056396857,],
	        vec![-0.10441319,-0.37278038,0.0748213,],
        ]);
        let exp = Matrix::from_vec2(vec![
	        vec![0.0, 0.0, 0.0,],
	        vec![0.0038859788,0.0 ,0.027655335,],
	        vec![0.0, 0.0,0.056396857,],
	        vec![0.0, 0.0, 0.0748213,],
        ]);
        let mut act = ReLu::new();
        act.forward(&data);
        assert_eq!(act.output, exp);
    }


    #[test]
    fn activation_leaky_relu() {
        let data = Matrix::from_vec2(vec![
	        vec![0.0,0.0,0.0,],
	        vec![0.0038859788,-0.09141435,0.027655335,],
	        vec![-0.005156948,-0.20069951,0.056396857,],
	        vec![-0.10441319,-0.37278038,0.0748213,],
        ]);
        let exp = Matrix::from_vec2(vec![
	vec![0.0,0.0,0.0],
	vec![0.0038859788,-0.0009141435,0.027655335],
	vec![-0.00005156948,-0.0020069951,0.056396857],
	vec![-0.0010441319,-0.0037278038000000003,0.0748213],
        ]);
        let mut act = LeakyReLu::new();
        act.forward(&data);
        assert_eq!(act.output, exp);
    }

    #[test]
    fn activation_sigmoid() {
        let data = Matrix::from_vec2(vec![
	        vec![0.0,0.0,0.0,],
	        vec![0.0038859788,-0.09141435,0.027655335,],
	        vec![-0.005156948,-0.20069951,0.056396857,],
	        vec![-0.10441319,-0.37278038,0.0748213,],
        ]);
        let exp = Matrix::from_vec2(vec![
	vec![ 0.5, 0.5, 0.5],
	vec![ 0.5009714934774704, 0.4771623140376805, 0.5069133931319145 ],
	vec![ 0.4987107658571685, 0.4499928684121575, 0.5140954784350359 ],
	vec![ 0.4739203917683193, 0.4078693549674623, 0.5186966034946126 ],
        ]);
        let mut act = Sigmoid::new();
        act.forward(&data);
        assert_eq!(act.output, exp);
    }

    #[test]
    fn activation_softmax() {
        let data = Matrix::from_vec2(vec![vec![ 0.        ,  0.        ],
             vec![ 0.11061079,  0.01053243],
             vec![ 0.22218466, -0.00408555],
             vec![ 0.26955076, -0.19609564],
        ]);
        let mut dense1 = Dense::new(2, 3);
        let mut relu = ReLu::new();
        let mut dense2 = Dense::new(3,3);
        let mut softmax = Softmax::new();
        dense1.forward(&data);
        relu.forward(&dense1.outputs);
        dense2.forward(&relu.output);
        softmax.forward(&dense2.outputs);
        let v = softmax.output;
        assert!(4.0 - v.sum() < 0.01);
        

    }
}

