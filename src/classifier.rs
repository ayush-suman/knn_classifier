pub trait Classifier<D, L> {
    fn learn(&mut self, data: D, label: L);
    fn predict(&self, data: D) -> L;
}


