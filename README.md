Welcome to the chesvm package. This is the source code of my project work - Maritime Objects Classification using Exemplar-SVMs - for my master degree in [TUHH](https://www.tuhh.de/tuhh/startseite.html). This work is inspried by [exemplar-SVM](https://github.com/quantombone/exemplarsvm) from [Tomasz Malisiewicz](http://www.cs.cmu.edu/~tmalisie/). The code is written in Matlab.

----

# Quick Start

The code has been tested on ubuntu 12.04 LTS.  Pre-compiled Mex files are included. If you are going to run on different systems, please run `features/features_compile.m` and `lib/libsvm-3.11/matlab/make.m` to compile. You may also need to install [the MatConcNet library](http://www.vlfeat.org/matconvnet/install/).

To run the package, you need to have image set and pre-trained CNN model downloaded from [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/). In general, you may use any pre-trained CNNs provided. For this project, I use [`imagenet-vgg-m`](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat). The downloaded CNN should be put into a `pretrained_cnn` folder under root directory of the package. The image set should aslo be under root directory and organized as the following:

```
<image_set>
├── <class_1>
│   ├── test
│   └── train
├── <class_2>
│   ├── test
│   └── train
└── <class_3>
    ├── test
    └── train
```


