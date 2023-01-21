use anyhow::Result;
use log::{debug, error, info};
use pytorch_tch::run_adam;

fn main() -> Result<()> {
    env_logger::init();
    debug!("debug!");
    error!("error!");
    info!("info!");
    println!("println!");
    dbg!(std::process::id());
    dbg!(std::env::temp_dir());
    let m = tch::vision::mnist::load_dir("data")?;
    let model = run_adam(&m, 50)?;

    println!("{:#?}", model);
    Ok(())
}
