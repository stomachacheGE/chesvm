function default_params = esvm_get_default_params

% All parameters should be set correctly here in order to run the package.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training/Mining parameters %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
default_params.training_params.train_max_mine_iterations = 100;
default_params.training_params.train_max_mined_images = 2500;
%The maximum number of negatives to keep in the cache while training.
default_params.training_params.train_max_negatives_in_cache = 2000;
%Maximum number of violating images before SVM is trained with current cache
default_params.training_params.train_max_images_per_iteration = 200;
%The constant which tells us the weight in front of the positives
%during SVM learning
default_params.training_params.train_positives_constant = 60;
default_params.training_params.train_svm_c = -3; %% regularize with 2 ^(-3);
%The svm update equation
default_params.training_params.training_function = @esvm_update_svm;

%Mining Queue mode can be one of:
% {'onepass','cycle-violators','front-violators'}
%
% onepass: a single pass through the mining queue
% cycle-violators: discard non-firing images, and place violators
%     (images with detections) at the end of the mining queue
% front-violators: same as above but place violators at front of
% queue
% The last two modes require a termination condition such as
% (train_max_mined_images) so that learning doesn't loop
% indefinitely
default_params.training_params.queue_mode = 'onepass';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Dataset parameters %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parent folder of all the image sets. 
% Make sure your training set is of the same
% structure, i.e. each class folder is split to 'train' and 'test' folders
default_params.datasets_params.img_folder = 'img'; 
% The image set used for training.
default_params.datasets_params.dataset_dir  = 'maritime_cropped';
% Dateset file type
default_params.datasets_params.file_ext = 'jpg';
% Result folder. The folder can have any valid folder name.
default_params.datasets_params.results_folder = 'results_maritime';
% The package assumes that the image files are sorted and named 
% with format '%06d.<file_type>'. If this is not the case for your
% imageset, firstly set the rename to 'true' so that the package will
% rename the image files according to that format. After the first run,
% you may set this parameter to 'false'.
default_params.datasets_params.rename = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Features parameters %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
default_params.features_params.cnn_params.feat_name = 'cnn'; 
default_params.features_params.cnn_params.model_folder = 'pretrained_cnn';
default_params.features_params.cnn_params.model_name = 'imagenet-vgg-m.mat';
% Output of this layer of the ConvNet will be taken as CNN feature
default_params.features_params.cnn_params.layer = 'relu6'; 

default_params.features_params.hog_params.feat_name = 'hog'; 
% Cell size used for extracting HoG feature. e.g., 8 * 8 pixels
default_params.features_params.hog_params.sbin = 8;
% Normalize the images before extracting HoG features.
default_params.features_params.hog_params.width = 150;
default_params.features_params.hog_params.height = 100;
default_params.features_params.hog_extractor = @esvm_extract_hog_feature;
