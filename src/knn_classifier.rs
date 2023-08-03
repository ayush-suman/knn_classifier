use std::collections::HashMap;
use crate::classifier::Classifier;
use crate::point::Point;

pub struct KNNClassifier {
    pub k: usize,
    data: Vec<Point>,
    labels: Vec<u8>,
}

impl KNNClassifier {
    pub fn new(k: usize, mut data: Vec<Vec<u8>>, labels: Vec<u8>) -> KNNClassifier {
        let mut points: Vec<Point> = Vec::new();

        let len = data.len();

        for _ in 0..len {
            let v = data.remove(0);
            points.push(Point{coordinates: v});
        }

        let nearest_neighbor = KNNClassifier {
            k,
            data: points,
            labels,
        };
        nearest_neighbor
    }

    pub fn add_points(&mut self, data: Vec<Point>, labels: Vec<u8>) {
        self.data.extend(data);
        self.labels.extend(labels);
    }
}

impl Classifier<Point, u8> for KNNClassifier {
    fn learn(&mut self, data: Point, label: u8) {
        self.data.push(data);
        self.labels.push(label);
    }

    fn predict(&self, data: &Point) -> u8 {
        let mut nearest_neighbors: Vec<(u128, u8)> = Vec::new();

        for (i, d) in self.data.iter().enumerate() {
            let distance = data.distance(d);
            let result = nearest_neighbors.binary_search_by(|probe| probe.0.cmp(&distance));
            match result {
                Ok(index) => nearest_neighbors.insert(index, (distance, self.labels[i])),
                Err(index) => nearest_neighbors.insert(index, (distance, self.labels[i]))
            }
        }

        let mut map: HashMap<u8, u64> = HashMap::new();
        let mut max_count: u64 = 0;
        let mut label = 0;
        let smallest = nearest_neighbors[0].0;

        for (i, neighbor) in nearest_neighbors.iter().enumerate() {
            if i >= self.k && neighbor.0 != smallest {
                println!("Breaking from loop at {}", &i);
                break;
            }

            let count = map.entry(neighbor.1).and_modify(|counter| *counter += 1).or_insert(1);

            if max_count < *count {
                max_count = *count;
                label = neighbor.1;
            }
        }

        return label;
    }
}
