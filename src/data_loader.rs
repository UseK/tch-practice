use anyhow::Result;

use crate::config::MY_DEVICE;
use serde::Deserialize;
use tch::{
    vision::{dataset::Dataset, image},
    Kind, Tensor,
};

pub fn load_datasets<T, F>(
    image_dir: T,
    annotation_path: T,
    transform: Option<F>,
) -> Result<Dataset>
where
    T: AsRef<std::path::Path>,
    F: Fn(Tensor) -> Tensor,
{
    let train_images = load_dir_and_transform(&image_dir, &transform)?.to_device(MY_DEVICE);
    let test_images = load_dir_and_transform(&image_dir, &transform)?.to_device(MY_DEVICE);
    let train_labels = load_annotations_with_no_header(&annotation_path)?.to_device(MY_DEVICE);
    let test_labels = load_annotations_with_no_header(&annotation_path)?.to_device(MY_DEVICE);

    Ok(Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 2, //TODO: calc from labels
    })
}

fn load_dir_and_transform<T, F>(image_dir: T, transform: &Option<F>) -> Result<Tensor>
where
    T: AsRef<std::path::Path>,
    F: Fn(Tensor) -> Tensor,
{
    const SIZE: i64 = 512;
    let tensor = image::load_dir(&image_dir, SIZE, SIZE)?;
    if let Some(f) = transform {
        Ok(f(tensor))
    } else {
        Ok(tensor)
    }
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

pub trait ToDevice {
    fn to_device(&self, device: tch::Device) -> Self;
}

impl ToDevice for Dataset {
    fn to_device(&self, device: tch::Device) -> Self {
        Self {
            train_images: self.train_images.to_device(device),
            train_labels: self.train_labels.to_device(device),
            test_images: self.test_images.to_device(device),
            test_labels: self.test_labels.to_device(device),
            labels: self.labels,
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::data_loader::{load_annotations_with_header, load_annotations_with_no_header};
    use std::{
        hash::{Hash, Hasher},
        io::Write,
    };
    use tch::{vision::mnist, Tensor};

    #[test]
    fn test_mnist() {
        let mnist = mnist::load_dir("data").unwrap();
        assert_eq!(mnist.train_images.size(), &[60000, 28 * 28]);
        assert_eq!(mnist.train_labels.size(), &[60000]);
        assert_eq!(mnist.test_images.size(), &[10000, 28 * 28]);
        assert_eq!(mnist.test_labels.size(), &[10000]);
        assert_eq!(mnist.labels, 10);
    }

    #[test]
    fn test_load_annotations() {
        let filename = get_temp_file(b"filename,label\n000001.jpg,0\n000002.jpg,1");
        let tensor = load_annotations_with_header(&filename).unwrap();
        assert_eq!(tensor, Tensor::of_slice(&[0, 1]));
    }

    #[test]
    fn test_load_annotations_with_no_header() {
        let filename = get_temp_file(b"000001.jpg,0\n000002.jpg,1");
        let tensor = load_annotations_with_no_header(&filename).unwrap();
        assert_eq!(tensor, Tensor::of_slice(&[0, 1]));
    }

    fn get_temp_file(buf: &[u8]) -> std::path::PathBuf {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        buf.hash(&mut hasher);
        let filename = std::env::temp_dir().join(format!(
            "annotation-{}-{}.csv",
            std::process::id(),
            hasher.finish()
        ));
        {
            let mut file = std::fs::File::create(&filename).unwrap();
            file.write_all(buf).unwrap();
        }
        filename
    }
}
