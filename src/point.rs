pub struct Point {
    pub(crate) coordinates: Vec<u8>
}

impl Point {
    pub fn distance(&self, other: &Point) -> u128 {
        let mut distance: u128 = 0;
        for (d1, d2) in self.coordinates.iter().zip( other.coordinates.iter()) {
            distance += (d1.abs_diff(d2.clone()) as u32).pow(2) as u128;
        }
        distance
    }
}
