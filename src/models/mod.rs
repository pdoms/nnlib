pub mod example;

use std::fs;

pub fn parse_csv(file: &str) -> (Vec<Vec<f64>>, Vec<i32>) {
    let contents = fs::read_to_string(file).expect("could not read the file");
    let mut y_data = Vec::<i32>::new();
    let x_data: Vec<Vec<f64>> = contents.split('\n')
        .filter(|x| x.len() > 0)
        .map(|line| {
        let data: Vec<&str> = line.splitn(3, ',').collect();
        let x: Vec<f64> = vec![data[0].parse().unwrap(), data[1].parse().unwrap()];
        y_data.push(data[2].parse().unwrap());
        x
    }).collect();
    (x_data, y_data)
}
