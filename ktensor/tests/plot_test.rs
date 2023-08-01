use plotters::prelude::*;
use std::fs::create_dir;

fn draw_scatter_plot(data: &[(f64, f64)], file_name: &str, min_value: f64, max_value: f64) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_margin = (max_value - min_value) * 0.05;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            -0.1f64..1.1f64,
            (min_value - y_margin)..(max_value + y_margin)
        )?;

    chart.configure_mesh().draw()?;

    let shape_style = ShapeStyle::from(&BLUE).filled();
    chart.draw_series(data.iter().map(|&(x, y)| Circle::new((x, y), 5, shape_style)))?;

    root.present()?;

    Ok(())
}

#[test]
fn scatter_plot_test() {
    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    let data = vec![(0.1, 0.2), (0.3, 0.4), (0.5, 0.7), (0.8, 0.6)];
    let filename = "output/scatter_plot.png";
    draw_scatter_plot(&data, filename, 0.0, 1.0).unwrap();
}

#[test]
fn toy_dataset_plot() {
    use ktensor::tensor::TensorRng;

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    let mut rng = TensorRng::new();

    let x = rng.gen::<f64, _>([100, 1]);
    let y
        = &x * 2.0 + 5.0
            + rng.gen::<f64, _>([100, 1]);

    let filename = "output/toy_dataset_plot.png";

    let x_data = x.get_data().iter().map(|x| *x).collect::<Vec<f64>>();
    let y_data = y.get_data().iter().map(|y| *y).collect::<Vec<f64>>();
    let min_value = y_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_value = y_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let data = x_data.iter().zip(y_data.iter()).map(|(x, y)| (*x, *y)).collect::<Vec<(f64, f64)>>();
    draw_scatter_plot(&data, filename, min_value, max_value).unwrap();
}
