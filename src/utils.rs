use std::fs::File;
use std::io::{BufReader, Read};
use crate::classifier::Classifier;
use crate::constants::IMAGE_SIZE;

pub async fn get_data_and_labels(file_name: &str) -> (Vec<Vec<u8>>, Vec<u8>) {
    let reader = BufReader::new(File::open(file_name).unwrap());

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();

    for (i, byte) in reader.bytes().enumerate() {
        let byte = byte.unwrap();
        if i % (IMAGE_SIZE + 1) == 0 {
            labels.push(byte);
            data.push(Vec::new());
        } else {
            data[i / (IMAGE_SIZE + 1)].push(byte);
        }
    }
    (data, labels)
}

pub async fn classify_and_count<T>(classifier: T, data: Vec<Vec<u8>>, labels: Vec<u8>) -> u32
where
    T: Classifier<Vec<u8>> + Sync
{

    let mut correct = 0;
    for (i, d) in data.iter().enumerate() {
        let label = classifier.classify(d);
        if label == labels[i] {
            correct += 1;
        }
    }

    println!("Correct: {}", correct);
    correct
}


