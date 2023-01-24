use anyhow::Result;
use log::debug;
use serde::Deserialize;
use tch::{
    kind::Kind,
    nn,
    nn::{Module, OptimizerConfig},
    vision::dataset::Dataset,
    Device, Tensor,
};

use tch::vision::image;

pub fn perceptron_like(p: nn::Path, dim: i64) -> impl Module {
    let x1 = p.zeros("x1", &[dim]);
    let x2 = p.zeros("x2", &[dim]);
    nn::func(move |xs| {
        debug!("x1: {:#?}", x1);
        debug!("x2: {:#?}", x2);
        xs * &x1 + xs.exp() * &x2
    })
}

pub fn nn_seq_like(vs: &nn::Path, in_dim: i64, out_dim: i64) -> impl Module {
    const HIDDEN_NODES: i64 = 128;
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            in_dim,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, HIDDEN_NODES, out_dim, Default::default()))
}

pub fn run_gradient_descent(xs: &Tensor, ys: &Tensor, epochs: u32) -> impl Module {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = perceptron_like(vs.root(), 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for epoch in 1..=epochs {
        let loss = (my_module.forward(xs) - ys)
            .pow_tensor_scalar(2)
            .sum(Kind::Float);

        if epoch % 100 == 0 {
            println!("\nepoch: {}/{}", epoch, epochs);
            println!("loss: {:#?}", loss);
        }
        opt.backward_step(&loss);
    }
    my_module
}

pub fn run_adam(m: &Dataset, epochs: u32) -> Result<impl Module> {
    dbg!(m.labels);

    dbg!(m.train_images.kind());
    dbg!(m.train_labels.kind());
    dbg!(m.test_images.kind());
    dbg!(m.test_labels.kind());

    dbg!(m.train_images.dim());
    dbg!(m.train_images.size());
    let (_, in_dim) = m.train_images.size2()?;
    let vs = nn::VarStore::new(Device::Cpu);
    let net = nn_seq_like(&vs.root(), in_dim, m.labels);
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 0..epochs {
        let loss = net
            .forward(&m.train_images)
            .cross_entropy_for_logits(&m.train_labels);
        opt.backward_step(&loss);
        let test_accuracy = net
            .forward(&m.test_images)
            .accuracy_for_logits(&m.test_labels);
        if epoch % 10 == 0 {
            println!(
                "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
                epoch,
                f64::from(&loss),
                100. * f64::from(&test_accuracy),
            );
        }
    }
    Ok(net)
}

pub fn load_datasets<T>(image_dir: T, annotation_path: T) -> Result<Dataset>
where
    T: AsRef<std::path::Path>,
{
    const SIZE: i64 = 512;
    let train_images = image::load_dir(&image_dir, SIZE, SIZE)?;
    let test_images = image::load_dir(&image_dir, SIZE, SIZE)?;
    let train_labels = load_annotations_with_no_header(&annotation_path)?;
    let test_labels = load_annotations_with_no_header(&annotation_path)?;

    Ok(Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 2,
    })
}

#[derive(Debug, Deserialize)]
pub struct Annotation {
    pub filename: std::path::PathBuf,
    pub label: i64,
}

pub fn load_annotations_with_header<T>(annotations_path: T) -> Result<Tensor>
where
    T: AsRef<std::path::Path>,
{
    let reader = std::fs::File::open(annotations_path)?;
    let mut csv_reader = csv::Reader::from_reader(reader);
    let data: Vec<i64> = csv_reader
        .deserialize::<Annotation>()
        .map(|r| r.unwrap().label)
        .collect();
    Ok(Tensor::of_slice(&data).to_kind(Kind::Int64))
}

pub fn load_annotations_with_no_header<T>(annotations_path: T) -> Result<Tensor>
where
    T: AsRef<std::path::Path>,
{
    let reader = std::fs::File::open(annotations_path)?;
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(reader);
    let mut labels: Vec<i64> = vec![];
    for r in csv_reader.records() {
        let r: csv::StringRecord = r?;
        let label = r.get(1).unwrap();
        labels.push(label.parse::<i64>()?);
    }
    Ok(Tensor::of_slice(&labels).to_kind(Kind::Int64))
}

#[cfg(test)]
mod tests {
    use std::{
        hash::{Hash, Hasher},
        io::Write,
    };
    use tch::Tensor;
    use tch::{nn::Module, vision::dataset::Dataset};

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_run_gradient_descent() {
        init();
        let xs = Tensor::of_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let ys = Tensor::of_slice(&[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]);
        let trained = crate::run_gradient_descent(&xs, &ys, 200);
        let forwarded = trained.forward(&xs);
        let expected = Tensor::of_slice(&[
            0.5894472002983093,
            0.4965878681698622,
            0.39920678950967375,
            0.2998769086883365,
            0.19998866544379637,
            0.09999951477228386,
            0.0,
        ]);
        assert_eq!(forwarded, expected);
    }

    #[test]
    fn test_run_adam_with_mnist() {
        init();
        let m = tch::vision::mnist::load_dir("data").unwrap();
        crate::run_adam(&m, 1).unwrap();
    }

    #[test]
    fn test_run_adam_to_error_with_1_dimension() {
        init();
        let m = Dataset {
            train_images: Tensor::of_slice(&[0.0, 0.1, 0.2]),
            train_labels: Tensor::of_slice(&[0]),
            test_images: Tensor::of_slice(&[0]),
            test_labels: Tensor::of_slice(&[0]),
            labels: 99,
        };
        assert!(crate::run_adam(&m, 99).is_err());
    }

    #[test]
    fn test_run_adam_with_2_dimension() {
        let m = Dataset {
            train_images: Tensor::of_slice2(&[&[0.0f32, 0.1, 0.2], &[0.3, 0.4, 0.5]]),
            train_labels: Tensor::of_slice(&[0i64, 1]),
            test_images: Tensor::of_slice2(&[&[0.0f32, 0.1, 0.2], &[0.3, 0.4, 0.5]]),
            test_labels: Tensor::of_slice(&[0i64, 1]),
            labels: 99,
        };
        assert!(crate::run_adam(&m, 99).is_ok());
    }

    fn get_temp_file(buf: &[u8]) -> std::path::PathBuf {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        buf.hash(&mut hasher);
        let filename = std::env::temp_dir().join(format!(
            "pytorch-tch-{}-{}.csv",
            std::process::id(),
            hasher.finish()
        ));
        {
            let mut file = std::fs::File::create(&filename).unwrap();
            file.write_all(buf).unwrap();
        }
        filename
    }

    #[test]
    fn test_load_annotations() {
        let filename = get_temp_file(b"filename,label\n000001.jpg,0\n000002.jpg,1");
        let tensor = crate::load_annotations_with_header(&filename).unwrap();
        assert_eq!(tensor, Tensor::of_slice(&[0, 1]));
    }

    #[test]
    fn test_load_annotations_with_no_header() {
        let filename = get_temp_file(b"000001.jpg,0\n000002.jpg,1");
        let tensor = crate::load_annotations_with_no_header(&filename).unwrap();
        assert_eq!(tensor, Tensor::of_slice(&[0, 1]));
    }
}
