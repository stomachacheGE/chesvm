function cnn_imagenet_googlenet()
%CNN_IMAGENET_GOOGLENET  Demonstrates how to use GoogLeNet

run ../../matlab/vl_setupnn

cnn_dir = '/media/datadrive/Various/CNN/pretrained_cnn/';
% modelPath =  fullfile(cnn_dir,'imagenet-vgg-f.mat'); % 93%
% modelPath =  fullfile(cnn_dir,'imagenet-vgg-m.mat'); % 95%
% modelPath =  fullfile(cnn_dir,'imagenet-vgg-s.mat'); % 85%
% modelPath = fullfile(cnn_dir,'imagenet-caffe-ref.mat'); % 97% with extended images !! or 96% the same as BoW
% modelPath = fullfile(cnn_dir,'imagenet-matconvnet-vgg-verydeep-16.mat'); % here 99.5%!!
% modelPath = fullfile(cnn_dir,'imagenet-matconvnet-vgg-f.mat'); %here 33%
% modelPath = fullfile(cnn_dir,'imagenet-matconvnet-vgg-m.mat'); %here 67%
modelPath = fullfile(cnn_dir,'imagenet-matconvnet-vgg-s.mat'); %here 88%
% modelPath = fullfile(cnn_dir,'imagenet-vgg-verydeep-16.mat');
% modelPath = fullfile(cnn_dir,'imagenet-googlenet-dag.mat'); 

% use_dag = true;
use_dag = false;

% modelPath = 'data/models/imagenet-googlenet-dag.mat' ;

if ~exist(modelPath)
  mkdir(fileparts(modelPath)) ;
  urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat', ...
    modelPath) ;
end

if ~use_dag
    net = load(modelPath);
else
    net = dagnn.DagNN.loadobj(load(modelPath)) ;
end

im = imread('peppers.png') ;
% im = imread('test8.jpg') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;

meanim = net.meta.normalization.averageImage;
if size(meanim,2) == 1 % gs Is a vector? Make image from the mean vector
    meanim = reshape(meanim, 1,1,3);
    meanim = repmat(meanim, net.meta.normalization.imageSize(1:2));
end
im_ = bsxfun(@minus,im_,meanim); % gs substract from average image
% im_ = im_ - net.meta.normalization.averageImage ;

if ~use_dag
    res = vl_simplenn(net, im_);
    scores = squeeze(gather(res(end).x)) ;
else
    net.eval({'data', im_}) ;
    scores = squeeze(gather(net.vars(end).value)) ;       
end

% show the classification result
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;
