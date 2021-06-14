# CNNBaseHashing

## Requirements

You need [Pytorch](https://pytorch.org/) 1.0+ to run this. The code has been tested with Python 3.6+ and runs both on MacOS 10.14.5 and Windows 10.

## Model

This folder consist of the structure of the deep hashing network.


## Network-parameters-and-options

This folder contains the options and parameters file of the deep hashing network, where the network parameter is the results after 300 epochs of training.


## Data

If you want to test the model, please obey the following rules of data directory structure:
```
<data_root>/
  test/
    val_class/
      test_image1.jpg
      test_image2.jpg
      ...
```

```val_class``` folders are so that we can use the standard torchvision data loaders without change.

## Running

You will need to install the requirements, then run 
```
python Hash_generation.py -o <network_option_root> -c <network_parameters_root> -s <test_image_root> -b <batchsize>
``` 
```batchsize``` here denotes how many images to generate hashes simultaneously.


## For example
```
python Hash_generation.py -o ./Network-parameters-and-options/Network-Options.pickle -c ./Network-parameters-and-options/Network-Parameters.pyt -s ./test_image -b 10
```

# Note 

if you don't have the NVDIA GPU, please add “map_location='cpu'” behind the "checkpoint = torch.load(args.checkpoint_file）", i.e., "checkpoint = torch.load(args.checkpoint_file,map_location='cpu')" in the file of "Hash_generation.py" to run the hash generation on the CPU.


## Result

Finally we will obtain the file "HashResult-{time}.csv". The sequence of each line represents each images in the testing set in order. 



