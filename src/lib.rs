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
    view_size: [i64; 4],
    p_size: i64,
}

impl Net {
    pub fn new(vs: &nn::Path, dataset: &Dataset) -> Net {
        let img_s = dataset.train_images.size();
        let p_size = 64 * Self::processed_size(img_s[2]) * Self::processed_size(img_s[3]);
        assert_eq!(img_s.len(), 4);
        let conv1 = nn::conv2d(vs, img_s[1], 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs, p_size, 1024, Default::default());
        let fc2 = nn::linear(vs, 1024, dataset.labels, Default::default());
        Net {
            conv1,
            conv2,
            fc1,
            fc2,
            view_size: [-1, img_s[1], img_s[2], img_s[3]],
            p_size,
        }
    }

    fn processed_size(x: i64) -> i64 {
        let x = (x - 4) / 2;
        (x - 4) / 2
    }
}

impl nn::Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        debug!("{:#?}", xs);
        let tmp = xs.view(self.view_size).apply(&self.conv1);
        debug!("conv1: {:#?}", tmp);
        let tmp = tmp.max_pool2d_default(2);
        debug!("pool1: {:#?}", tmp);
        let tmp = tmp.apply(&self.conv2);
        debug!("conv2: {:#?}", tmp);
        let tmp = tmp.max_pool2d_default(2);
        debug!("pool2: {:#?}", tmp);
        debug!("p_size: {:#?}", self.p_size);
        debug!("-----------------");
        tmp.view([-1, self.p_size])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
    }
}

pub fn train<F, M>(
    dataset: &Dataset,
    module_f: F,
    opt_config: impl OptimizerConfig,
    device: Device,
    epochs: usize,
) -> Result<M>
where
    F: Fn(&nn::Path, &Dataset) -> M,
    M: Module,
{
    let vs = nn::VarStore::new(device);
    let mut opt = opt_config.build(&vs, 1e-3)?;
    let module = module_f(&vs.root(), dataset);
    for epoch in 0..epochs {
        let loss = module
            .forward(&dataset.train_images)
            .cross_entropy_for_logits(&dataset.train_labels);
        opt.backward_step(&loss);
        let test_accuracy = module
            .forward(&dataset.test_images)
            .accuracy_for_logits(&dataset.test_labels);
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
    use crate::{
        config::MY_DEVICE,
        data_loader::{gen_random_dataset, ToDevice},
        nn_seq_like,
    };

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
    fn test_cnn() {
        for labels in 1..=4 {
            let vs = VarStore::new(tch::Device::Cpu);
            let dataset = gen_random_dataset(&[22, 1, 28, 28], &[33, 1, 28, 28], labels);
            let model = super::Net::new(&vs.root(), &dataset);
            let forwarded = model.forward(&dataset.test_images);
            assert_eq!(
                forwarded.size(),
                &[dataset.test_images.size()[0], dataset.labels]
            );
            assert_eq!(forwarded.kind(), Kind::Float);
        }
    }

    #[test]
    fn test_train() -> anyhow::Result<()> {
        let m = gen_2dim_dataset();
        let (_, in_dim) = m.train_images.size2().unwrap();
        let model = train(
            &m,
            |vs, dataset| nn_seq_like(vs, in_dim, dataset.labels),
            tch::nn::Adam::default(),
            tch::Device::Cpu,
            50,
        )?;
        let forwarded = model.forward(&m.test_images);
        assert_eq!(forwarded.size(), &[m.test_images.size()[0], m.labels]);
        assert_eq!(forwarded.kind(), Kind::Float);
        Ok(())
    }

    #[test]
    fn test_train_cnn() -> anyhow::Result<()> {
        let dataset = gen_random_dataset(&[2, 3, 20, 20], &[3, 3, 20, 20], 77);
        let model = train(
            &dataset,
            crate::Net::new,
            tch::nn::Adam::default(),
            tch::Device::Cpu,
            5,
        )?;
        let forwarded = model.forward(&dataset.test_images);
        assert_eq!(
            forwarded.size(),
            &[dataset.test_images.size()[0], dataset.labels]
        );
        assert_eq!(forwarded.kind(), Kind::Float);
        Ok(())
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
