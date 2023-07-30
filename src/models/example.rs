use super::parse_csv;
use crate::layer::dense::Dense;
use crate::activation::{Activation, ReLu};

pub fn example_model(file: &str) {
    let (x_data, y_data) = parse_csv(file);
    let d1 = Dense::new(2, 3);
    let a1 = ReLu::new();
    let d2 = Dense::new(3,3);
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn models_example() {
        example_model("src/models/spiral_data.csv");
    }
}
