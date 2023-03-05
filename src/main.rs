use anyhow::Result;
use log::info;
use tch_practice::{train, Net};

fn main() -> Result<()> {
    run()
}

fn run() -> Result<()> {
    env_logger::init();
    // let m = tch::vision::mnist::load_dir("data")?;
    let m = tch_practice::data_loader::load_datasets(
        "data/mydata/images",
        "data/mydata/annotation_0_1.csv",
        // Some(|t: tch::Tensor| t.flatten(1, 3).internal_cast_float(true)),
        Some(|t: tch::Tensor| t.internal_cast_float(true)),
    )?;
    // let model = tch_practice::run_adam(&m, 2)?;

    let model = train(&m, Net::new, tch::nn::Adam::default(), tch::Device::Cpu, 20);

    info!("{:#?}", model);
    Ok(())
}
