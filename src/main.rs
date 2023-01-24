use anyhow::Result;
use log::{debug, error, info};

fn main() -> Result<()> {
    env_logger::init();
    debug!("debug!");
    error!("error!");
    info!("info!");
    println!("println!");
    run()
}

fn run() -> Result<()> {
    // let m = tch::vision::mnist::load_dir("data")?;
    let m = pytorch_tch::load_datasets("data/mydata/images", "data/mydata/annotation_0_1.csv")?;
    let model = pytorch_tch::run_adam(&m, 50)?;

    println!("{:#?}", model);
    Ok(())
}
