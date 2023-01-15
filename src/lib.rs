use anyhow::Result;
use log::debug;
use tch::{
    kind,
    nn::{self, Module, OptimizerConfig},
    vision::dataset::Dataset,
    Device, Tensor,
};

pub fn perceptron_like(p: nn::Path, dim: i64) -> impl nn::Module {
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

pub fn run_gradient_descent(xs: &Tensor, ys: &Tensor, epochs: u32) -> impl nn::Module {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = perceptron_like(vs.root(), 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for epoch in 1..=epochs {
        let loss = (my_module.forward(xs) - ys)
            .pow_tensor_scalar(2)
            .sum(kind::Kind::Float);

        if epoch % 100 == 0 {
            println!("\nepoch: {}/{}", epoch, epochs);
            println!("loss: {:#?}", loss);
        }
        opt.backward_step(&loss);
    }
    my_module
}

const IMAGE_DIM: i64 = 784;
const LABELS: i64 = 10;

pub fn run_adam(m: &Dataset, epochs: u32) -> Result<impl nn::Module> {
    let vs = nn::VarStore::new(Device::Cpu);
    let net = nn_seq_like(&vs.root(), IMAGE_DIM, LABELS);
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 0..epochs {
        let loss = net
            .forward(&m.train_images)
            .cross_entropy_for_logits(&m.train_labels);
        opt.backward_step(&loss);
        let test_accuracy = net
            .forward(&m.test_images)
            .accuracy_for_logits(&m.test_labels);
        if epoch % 50 == 0 {
            println!(
                "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
                epoch + 1,
                f64::from(&loss),
                100. * f64::from(&test_accuracy),
            );
        }
    }
    Ok(net)
}

#[cfg(test)]
mod tests {
    use tch::nn::Module;
    use tch::Tensor;

    use crate::{run_adam, run_gradient_descent};

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_run_gradient_descent() {
        init();
        let xs = Tensor::of_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let ys = Tensor::of_slice(&[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]);
        let trained = run_gradient_descent(&xs, &ys, 200);
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
    fn test_run_adam() {
        init();
        let m = tch::vision::mnist::load_dir("data").unwrap();
        let _ = run_adam(&m, 1);
    }
}
