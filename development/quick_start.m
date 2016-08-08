
addpath(genpath(pwd));

% MatConvNet
mcnpath = fullfile('lib','matconvnet-1.0-beta18','matlab');
run(fullfile(mcnpath,'vl_setupnn.m'));

% load the pre-trained CNN
net = load('./pretrained_cnn/imagenet-vgg-m.mat') ;
%net = vl_simplenn_tidy(net) ;

% load and preprocess an image
im = imread('peppers.png') ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;

% run the CNN
res = vl_simplenn(net, im_) ;

% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;