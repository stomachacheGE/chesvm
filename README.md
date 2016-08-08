Welcome to the chesvm package. This is the source code of my project work - Maritime Objects Classification using Exemplar-SVMs - for my master degree in [TUHH](https://www.tuhh.de/tuhh/startseite.html). This work is inspried by [exemplar-SVM](https://github.com/quantombone/exemplarsvm) from [Tomasz Malisiewicz](http://www.cs.cmu.edu/~tmalisie/). The code is written in Matlab.

The advantage of using exemplar-SVMs is that it is able to retrieve images in the training set which look similar to the test image. Thus it can be used for meta-data (e.g., segmentation and 3D model) transfer, which could be beneficial for overall scene understanding.

----

# Quick Start

### Download source code

Simply clone the repo and `cd` to the directory.

```sh
$ git clone git://github.com/stomachacheGE/chesvm.git
$ cd chesvm
```

### Setup

The code has been tested on ubuntu 12.04 LTS.  Pre-compiled Mex files are included. If you are going to run on different systems, please run `features/features_compile.m` and `lib/libsvm-3.11/matlab/make.m` to compile. You may also need to install [the MatConcNet library](http://www.vlfeat.org/matconvnet/install/).

To run the package, you need to have image set and pre-trained CNN model downloaded from [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/). In general, you may use any pre-trained CNNs provided on that page. For this project, I use [`imagenet-vgg-m`](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat). The downloaded CNN should be put into a `pretrained_cnn` folder under root directory of the package. The image set should aslo be under root directory and organized as the following:

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

### Configure Parameters

Before actual training exemplar-SVMs, the parameters in file `esvm_get_default_params.m` should be set correctly or to adapt to your needs. For example, you may change the cell size used to extract HoG feature, or the CNN layer whose output is used as CNN feature.

 ```matlab
% Output of this layer of the ConvNet will be taken as CNN feature
default_params.features_params.cnn_params.layer = 'relu6';  
% Cell size used for extracting HoG feature. e.g., 8 * 8 pixels
default_params.features_params.hog_params.sbin = 8;
```

### Train models

After correct setups and configurations, you may start to train your models. The main interface used to train models is the function `esvm_train(feature, algorithm, calibration, hard_negative)`.

```sh
$ matlab
>> addpath(genpath(pwd))
>> esvm_train('hog','esvm',false,false);
```

`feature` could be 'hog', 'cnn' or 'cnnhog'. `algorithm` could be 'esvm' or 'svm'. `calibration` and `hard_negative` are boolean variables. 

If you want to train models with all available algorithms and features, simply run `esvm_train_all`.

```sh
$ matlab
>> addpath(genpath(pwd))
>> esvm_train_all;
```

### Graphical User Interface

![](https://github.com/stomachacheGE/chesvm/blob/master/GUI/GUI.png)

The package provides a GUI, with which users may play with trained models to test images. Before using the GUI, make sure at least some models are already trained and live in the result folder. Otherwise, the GUI is unable to make predictions. Start the GUI by following commands:

```sh
$ matlab
>> addpath(genpath(pwd))
>> GUI;
```

The GUI is quite straightforward once you open the interface. The `Try My Luck` button simply choose a random image in your test set and make predictions. 

Note if, by any chance, you want the images I used for my project and the trained models, you may contact me by [email](mailto:liangchengfu001@gmail.com).

