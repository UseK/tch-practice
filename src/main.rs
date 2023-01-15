use anyhow::Result;
use log::{debug, error, info};
use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Tensor};

fn my_module(p: nn::Path, dim: i64) -> impl nn::Module {
    let x1 = p.zeros("x1", &[dim]);
    let x2 = p.zeros("x2", &[dim]);
    nn::func(move |xs| {
        debug!("x1: {:#?}", x1);
        debug!("x2: {:#?}", x2);
        xs * &x1 + xs.exp() * &x2
    })
}

#[allow(dead_code)]
fn gradient_descent() {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = my_module(vs.root(), 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for _idx in 1..50 {
        let xs = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let ys = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let loss = (my_module.forward(&xs) - ys)
            .pow_tensor_scalar(2)
            .sum(kind::Kind::Float);
        opt.backward_step(&loss);
        println!("{:#?}", opt);
        println!("{:#?}", loss);
    }
}

#[allow(dead_code)]
fn gradient_descent_from_raw(xs: &Tensor, ys: &Tensor, epochs: u32) -> impl nn::Module {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = my_module(vs.root(), 7);
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
const HIDDEN_NODES: i64 = 128;
const LABELS: i64 = 10;

fn net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            IMAGE_DIM,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}

pub fn run() -> Result<()> {
    let m = tch::vision::mnist::load_dir("data")?;
    let vs = nn::VarStore::new(Device::Cpu);
    let net = net(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 0..200 {
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
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    debug!("debug!");
    error!("error!");
    info!("info!");
    run()
}

#[cfg(test)]
mod tests {
    use crate::gradient_descent_from_raw;
    use tch::{nn::Module, Tensor};

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test() {
        init();
        let xs = Tensor::of_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let ys = Tensor::of_slice(&[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]);
        let trained = gradient_descent_from_raw(&xs, &ys, 200);
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
}
