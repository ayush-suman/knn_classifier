pub trait Classifier<T> {
    fn classify(&self, data: &T) -> u8;
}


