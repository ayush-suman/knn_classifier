mod constants;
mod classifier;
mod knn_classifier;
mod utils;
mod point;

use tokio::sync::mpsc;

use crate::knn_classifier::KNNClassifier;
use crate::point::Point;
use crate::utils::{classify_and_count, get_data_and_labels};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut nearest_neighbor = KNNClassifier::new(10, Vec::new(), Vec::new());

    let (tx, mut rx) = mpsc::channel::<(Vec<Point>, Vec<u8>)>(4);
    let tx2 = tx.clone();
    let tx3 = tx.clone();
    let tx4 = tx.clone();

    tokio::spawn(async move {
        let (data, labels) = get_data_and_labels("./data/cifar-10/data_batch_1.bin").await;
        println!("Loaded data from data_batch_1.bin");
        tx.send((data, labels)).await.unwrap();
    });


    tokio::spawn(async move {
        let (data, labels) = get_data_and_labels("./data/cifar-10/data_batch_2.bin").await;
        println!("Loaded data from data_batch_2.bin");
        tx2.send((data, labels)).await.unwrap();
    });

    tokio::spawn(async move {
        let (data, labels) = get_data_and_labels("./data/cifar-10/data_batch_3.bin").await;
        println!("Loaded data from data_batch_3.bin");
        tx3.send((data, labels)).await.unwrap();
    });

    tokio::spawn(async move {
        let (data, labels) = get_data_and_labels("./data/cifar-10/data_batch_4.bin").await;
        println!("Loaded data from data_batch_4.bin");
        tx4.send((data, labels)).await.unwrap();
    });

    while let Some((data, label)) = rx.recv().await {
        nearest_neighbor.add_points(data, label);
    }

    drop(rx);

    let (data, labels) = get_data_and_labels("./data/cifar-10/data_batch_5.bin").await;
    println!("Loaded data from data_batch_5.bin");

    // let len = data.len();
    const SIZE: usize = 100;

    let correct = classify_and_count(nearest_neighbor, data, labels, SIZE).await;

    println!("Accuracy: {}%", correct as f64 / SIZE as f64 * 100.0);

    Ok(())
}
