pub mod config;
pub mod data_loader;
use anyhow::Result;
use config::MY_DEVICE;
use log::debug;
use tch::{
    kind::Kind,
    nn::{self, Module, OptimizerConfig},
    vision::dataset::Dataset,
    Device, Tensor,
};

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
    let vs = nn::VarStore::new(MY_DEVICE);
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

#[derive(Debug)]
pub struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    pub fn new(vs: &nn::Path) -> Net {
        let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs, 1024, 1024, Default::default());
        let fc2 = nn::linear(vs, 1024, 10, Default::default());
        Net {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

impl nn::Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.view([-1, 1, 28, 28])
            .apply(&self.conv1)
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .view([-1, 1024])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
    }
}

pub fn train<F, M>(
    m: &Dataset,
    module_f: F,
    opt_config: impl OptimizerConfig,
    device: Device,
    epochs: usize,
) -> Result<M>
where
    F: Fn(&nn::Path) -> M,
    M: Module,
{
    let vs = nn::VarStore::new(device);
    let mut opt = opt_config.build(&vs, 1e-3)?;
    let module = module_f(&vs.root());
    for epoch in 0..epochs {
        let loss = module
            .forward(&m.train_images)
            .cross_entropy_for_logits(&m.train_labels);
        opt.backward_step(&loss);
        let test_accuracy = module
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
    Ok(module)
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
    let vs = nn::VarStore::new(MY_DEVICE);
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

#[cfg(test)]
mod tests {
    use tch::{nn::Module, vision::dataset::Dataset};
    use tch::{nn::VarStore, Kind, Tensor};

    use super::train;
    use crate::{config::MY_DEVICE, data_loader::ToDevice, nn_seq_like};

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn tensor_assert_eq(left: &Tensor, right: &Tensor) {
        let delta = left - right;
        delta.iter::<f64>().unwrap().for_each(|v| {
            if v > 0.0000001 {
                panic!(
                    "panic in tensor_assert_eq\n left: {}\nright: {}",
                    left, right
                );
            }
        });
    }

    #[test]
    fn test_run_gradient_descent() {
        init();
        let xs = Tensor::of_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).to_device(MY_DEVICE);
        let ys = Tensor::of_slice(&[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]).to_device(MY_DEVICE);
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
        ])
        .to_device(MY_DEVICE);
        tensor_assert_eq(&forwarded, &expected);
    }

    #[test]
    fn test_run_adam_with_mnist() {
        init();
        let m = tch::vision::mnist::load_dir("data")
            .unwrap()
            .to_device(MY_DEVICE);
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
        let m = gen_2dim_dataset();
        assert!(crate::run_adam(&m, 99).is_ok());
    }

    #[test]
    fn test_train() -> anyhow::Result<()> {
        let m = gen_2dim_dataset();

        let (_, in_dim) = m.train_images.size2().unwrap();
        let model = train(
            &m,
            |vs| nn_seq_like(vs, in_dim, m.labels),
            tch::nn::Adam::default(),
            tch::Device::Cpu,
            50,
        )?;
        let _ = model.forward(&m.test_images);
        Ok(())
    }

    #[test]
    fn test_cnn() {
        let vs = VarStore::new(tch::Device::Cpu);
        let m = gen_random_dataset();
        let model = super::Net::new(&vs.root());
        let forwarded = model.forward(&m.test_images);
        assert_eq!(forwarded.size(), &[1, 10]);
        assert_eq!(forwarded.kind(), Kind::Float);
    }

    fn gen_random_dataset() -> Dataset {
        Dataset {
            train_images: Tensor::rand(&[1, 1, 28, 28], (Kind::Float, MY_DEVICE)),
            train_labels: Tensor::rand(&[10], (Kind::Float, MY_DEVICE)),
            test_images: Tensor::rand(&[1, 1, 28, 28], (Kind::Float, MY_DEVICE)),
            test_labels: Tensor::rand(&[10], (Kind::Float, MY_DEVICE)),
            labels: 10,
        }
        .to_device(MY_DEVICE)
    }

    fn gen_2dim_dataset() -> Dataset {
        Dataset {
            train_images: Tensor::of_slice2(&[&[0.0f32, 0.1, 0.2], &[0.3, 0.4, 0.5]]),
            train_labels: Tensor::of_slice(&[0i64, 1]),
            test_images: Tensor::of_slice2(&[&[0.0f32, 0.1, 0.2], &[0.3, 0.4, 0.5]]),
            test_labels: Tensor::of_slice(&[0i64, 1]),
            labels: 99,
        }
        .to_device(MY_DEVICE)
    }
}
