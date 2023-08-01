mkdir data
curl -o ./data/cifer-10.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf ./data/cifer-10.tar.gz -C ./data
rm ./data/cifer-10.tar.gz
mv ./data/cifar-10-batches-bin ./data/cifar-10
