use super::parse_csv;
use crate::layer::Layer;
use crate::layer::dense::Dense;
use crate::activation::{Activation, ReLu};
use crate::loss::SoftmaxCatCrossEnt;
use crate::metrics::softmax_accuracy;

pub fn example_model(file: &str) {
    println!("Example Model:");
    let (x_data, y_data) = parse_csv(file);
    let mut d1 = Dense::new(2, 3);
    let mut a1 = ReLu::new();
    let mut d2 = Dense::new(3,3);
    let mut loss_activ = SoftmaxCatCrossEnt::new();
    d1.forward(&x_data);
    a1.forward(&d1.outputs);
    d2.forward(a1.output());
    let loss = loss_activ.forward(&d2.outputs, &y_data);
    println!("\tLoss: {loss}");
    println!("\tAcc: {}", softmax_accuracy(&loss_activ.output, &y_data));
    let dvals = loss_activ.output.clone();
    loss_activ.backward(&dvals, &y_data);
    d2.backward(&loss_activ.dinputs);
    a1.backward(d2.dinputs);
    d1.backward(&a1.dinputs);
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn models_example() {
        example_model("src/models/spiral_data.csv");
    }
}
