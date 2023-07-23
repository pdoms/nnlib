use crate::la::Matrix;

pub struct Dense {
    pub weights: Matrix,
    pub biases: Matrix,
    pub inputs: Matrix,
    pub outputs: Matrix
}

impl Dense {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        Self {
            weights: Matrix::random(n_inputs, n_neurons, -1.0..1.0),
            biases: Matrix::zeroed(0, n_neurons),
            inputs: Matrix::new(1,1),
            outputs: Matrix::new(1,1),
        }
    }

    pub fn forward(&mut self, inputs: Matrix) {
        self.inputs = inputs.clone();
        let dot = self.inputs.dot(&self.weights);
        dot.add_vector(&self.biases);
        self.outputs = dot.clone()
    }
}

#[cfg(test)]
mod test {

    use super::*;
    
    #[test]
    fn dense_forward() {
        //this will not assert but print
        let data = Matrix::from_vec2(vec![vec![ 0.        ,  0.        ],
             vec![ 0.11061079,  0.01053243],
             vec![ 0.22218466, -0.00408555],
             vec![ 0.26955076, -0.19609564],
        ]);
        
        let mut layer = Dense::new(2, 3);
        layer.forward(data);
        let outs = layer.outputs;
        mat_dump!(outs);
    }
}
