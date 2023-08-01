# KNN Classifier in Rust

This is an example of KNN Classifier built in Rust. To run the example, follow the steps below:
1. Clone the repository (The obvious)
2. Run the script `./init.sh` to download the CIFAR-10 dataset, which is used in this example. 
3. Run `cargo run` in the root directory of the project.

## Performance as an Image Classifier
For image classification task on CIFAR-10 dataset, the KNN Classifier achieves an accuracy of ~ 0.4 on the test set. The accuracy is not very high, but it is not bad for a simple classifier like KNN. The accuracy can be improved by using more advanced classifiers like SVM, or by using deep learning models like CNN.

