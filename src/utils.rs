use std::cmp::min;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::Arc;
use tokio::sync::mpsc;
use crate::classifier::Classifier;
use crate::constants::{DATA_SIZE, IMAGE_SIZE};
use crate::point::Point;

pub async fn get_data_and_labels(file_name: &str) -> (Vec<Point>, Vec<u8>) {
    let reader = BufReader::new(File::open(file_name).unwrap());

    let mut data: Vec<Point> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();

    for (i, byte) in reader.bytes().enumerate() {
        let byte = byte.unwrap();
        let mut point_coordinates: Vec<u8> = Vec::new();
        if i % (IMAGE_SIZE + 1) == 0 {
            labels.push(byte);

        } else {
            point_coordinates.push(byte);
            if i % IMAGE_SIZE + 1 == IMAGE_SIZE {
                data.push(Point { coordinates: point_coordinates.clone() });
                point_coordinates.clear();
            }
        }
    }
    (data, labels)
}

pub async fn classify_and_count<T>(classifier: T, mut data: Vec<Point>, mut labels: Vec<u8>, size: usize) -> u32
where T: Classifier<Point, u8> + Sync + Send + 'static {

    let mut correct = 0;
    let (tx, mut rx) = mpsc::channel::<u32>(size);
    let len = data.len();
    let arc_classifier = Arc::new(classifier);

    for i in 0..min(len, size) {
        let d = data.remove(0);
        let l = labels.remove(0);

        let cloned_tx = tx.clone();
        let cl = Arc::clone(&arc_classifier);

        tokio::spawn(async move {
            let label = cl.predict(&d);
            if label == l {
                cloned_tx.send(1).await.unwrap();
            } else {
                cloned_tx.send(0).await.unwrap();
            }
            drop(cl);
        });
    }


    let mut i = 0;
    while let Some(r) = rx.recv().await {
        correct += r;
        i += 1;
        if i % size == 0 {
            break;
        }
    }

    drop(rx);

    println!("Correct: {}", correct);
    correct
}


