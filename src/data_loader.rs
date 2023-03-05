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

pub fn gen_random_dataset(train_size: &[i64; 4], test_size: &[i64; 4], labels: i64) -> Dataset {
    Dataset {
        train_images: Tensor::rand(train_size, (Kind::Float, MY_DEVICE)),
        train_labels: Tensor::randint(labels, &[train_size[0]], (Kind::Int64, MY_DEVICE)),
        test_images: Tensor::rand(test_size, (Kind::Float, MY_DEVICE)),
        test_labels: Tensor::randint(labels, &[test_size[0]], (Kind::Int64, MY_DEVICE)),
        labels,
    }
    .to_device(MY_DEVICE)
}

#[cfg(test)]
mod tests {
    use crate::data_loader::{
        gen_random_dataset, load_annotations_with_header, load_annotations_with_no_header,
    };
    use std::{
        hash::{Hash, Hasher},
        io::Write,
    };
    use tch::{vision::mnist, Kind, Tensor};

    #[test]
    fn test_mnist() {
        let mnist = mnist::load_dir("data").unwrap();
        assert_eq!(mnist.train_images.size(), &[60000, 28 * 28]);
        assert_eq!(mnist.train_labels.size(), &[60000]);
        assert_eq!(mnist.test_images.size(), &[10000, 28 * 28]);
        assert_eq!(mnist.test_labels.size(), &[10000]);
        assert_eq!(mnist.labels, 10);

        assert_eq!(mnist.train_images.kind(), Kind::Float);
        assert_eq!(mnist.test_images.kind(), Kind::Float);

        assert_eq!(mnist.train_labels.kind(), Kind::Int64);
        assert_eq!(mnist.test_labels.kind(), Kind::Int64);
    }

    #[test]
    fn test_gen_random_dataset() {
        let dataset = gen_random_dataset(&[55, 1, 28, 28], &[66, 1, 28, 28], 77);
        println!(
            "{:#?}",
            dataset
                .train_labels
                .iter::<i64>()
                .unwrap()
                .collect::<Vec<i64>>()
        );
        assert_eq!(dataset.train_images.size(), &[55, 1, 28, 28]);
        assert_eq!(dataset.train_labels.size(), &[55]);
        assert_eq!(dataset.test_images.size(), &[66, 1, 28, 28]);
        assert_eq!(dataset.test_labels.size(), &[66]);
        assert_eq!(dataset.labels, 77);

        assert_eq!(dataset.train_images.kind(), Kind::Float);
        assert_eq!(dataset.test_images.kind(), Kind::Float);

        assert_eq!(dataset.train_labels.kind(), Kind::Int64);
        assert_eq!(dataset.test_labels.kind(), Kind::Int64);
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
