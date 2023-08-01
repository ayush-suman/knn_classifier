use std::collections::HashMap;
use std::iter::zip;
use crate::classifier::Classifier;

pub struct KNNClassifier {
    pub k: u8,
    data: Vec<Vec<u8>>,
    labels: Vec<u8>,
}

impl KNNClassifier {
    pub fn new(k: u8, data: Vec<Vec<u8>>, labels: Vec<u8>) -> KNNClassifier {
        let nearest_neighbor = KNNClassifier {
            k,
            data,
            labels,
        };
        nearest_neighbor
    }

    pub fn add_point(&mut self, data: Vec<u8>, label: u8) {
        self.data.push(data);
        self.labels.push(label);
    }

pub fn add_points(&mut self, data: Vec<Vec<u8>>, labels: Vec<u8>) {
        self.data.extend(data);
        self.labels.extend(labels);
    }

    pub fn distance(&self, data1: &Vec<u8>, data2: &Vec<u8>) -> u64 {
        let mut distance: u64 = 0;
        for (d1, d2) in zip(data1.iter(), data2.iter()) {
            distance += (*d1 as u64).abs_diff(*d2 as u64);
        }
        distance
    }
}

impl Classifier<Vec<u8>> for KNNClassifier {
    fn classify(&self, data: &Vec<u8>) -> u8 {
        let mut nearest_neighbors: Vec<(u64, u8)> = Vec::new();

        for (i, d) in self.data.iter().enumerate() {
            let distance = self.distance(data, d);
            nearest_neighbors.push((distance, self.labels[i]));
            println!("{} {} {}", i, self.labels[i], distance)
        }

        nearest_neighbors.sort_by(|a, b| a.0.cmp(&b.0));

        println!("Distance sorted");

        let mut map: HashMap<u8, u8> = HashMap::new();
        let mut max_count: u8 = 0;
        let mut max_index = 0;

        for neighbor in nearest_neighbors {
            let count = map.entry(neighbor.1).or_insert(0);
            *count += 1;
            if *count == self.k {
                println!("Found the k nearest neighbors");
                return neighbor.1;
            }
            if max_count < *count {
                max_count = *count;
                max_index = neighbor.1;
            }
        }

        println!("Found max nearest neighbors");
        return max_index;
    }
}
