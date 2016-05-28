Matlab Script for Image Classification Tasks
=============================================

The example should illustrate the usage of pretrained convolutional neural network (CNN) model as automatic feature extractor in combination with SVM classifier.

Execution
---------

Run in Matlab run_CNN_SVM.m to perform classification on images placed in 'img' folder. 
Follow the code comments for each processing step.

If MatConvNet lib produces errors. 
First you need to compile the that lib.
In Matlab change to folder ../lib/matconvnet-1.0-beta18/matlab
and run 
>> vl_compilenn
If compilation produces folowing error:
fatal error: jpeglib.h: No such file or directory 
Install libjpeg-dev
>> sudo apt-get install libjpeg-dev
Compile again and check if running 
>> vl_setupnn 
runs silent.
