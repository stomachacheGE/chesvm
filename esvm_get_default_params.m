function default_params = esvm_get_default_params

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training/Mining parameters %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

default_params.training_params.train_max_mine_iterations = 100;

%Maximum TOTAL number of image accesses from the mining queue
default_params.training_params.train_max_mined_images = 2500;


%NOTE: I don't think these fields are being used since I set the
%global detection threshold to -1.
%when mining, we keep the N negative support vectors as well as
%some more beyond the -1 threshold (alpha*N), but no more than
%1000, where alpha is the "keep nsv multiplier"
default_params.training_params.train_keep_nsv_multiplier = 3;

%The maximum number of negatives to keep in the cache while training.
default_params.training_params.train_max_negatives_in_cache = 2000;

%Maximum number of violating images before SVM is trained with current cache
%default_params.training_params.train_max_images_per_iteration = 400;
default_params.training_params.train_max_images_per_iteration = 50;

%The constant which tells us the weight in front of the positives
%during SVM learning
default_params.training_params.train_positives_constant = 60;

%ICCV11 constant for SVM learning
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
default_params.training_params.hog_extractor = @esvm_extract_hog_feature;

% if non-zero, sets weight of positives such that positives and
%negatives are treated equally
%default_params.BALANCE_POSITIVES = 0;

% % NOTE: this stuff is experimental and currently disabled (see
% % do_svm.m). The goal was to perform dimensionality reduction
% % before the learning process.
% % If non-zero, perform learning in dominant-gradient space
% default_params.DOMINANT_GRADIENT_PROJECTION = 0;
% % The dimensionality of the local max-gradient descriptor
% default_params.DOMINANT_GRADIENT_PROJECTION_K = 2;
% % If enabled, do PCA on training data right before SVM (this
% % automatically converts the result to a descriptor in the RAW feature
% % space)
% default_params.DO_PCA = 0;
% % The degree of the PCA.esvm_get_datasets_info
% default_params.PCA_K = 300;
% % If enabled, only do PCA from the positives (so the subspace is what
% % spans the positive examples)
% default_params.A_FROM_POSITIVES = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Dataset parameters %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
default_params.datasets_params.img_folder = 'img'; % parent folder of all the data sets
% Examplary Caltech dataset. Make sure your training set is of the same
% structure, i.e. each class folder is split to 'train' and 'test' folders
%default_params.datasets_params.dataset_dir = '4_ObjectCategories_test_train'; % examplary Caltech dataset
%default_params.datasets_params.dataset_dir  = 'vais_export_EO_pair_03-03'; % VAIS EO images from pairs
% dataset_dir = 'vais_export_IR_pair_03-03'; % VAIS IR images from pairs
%default_params.datasets_params.dataset_dir  = 'VOC_2007_cropped';
default_params.datasets_params.dataset_dir  = 'maritime_cropped';
% Dateset file type
default_params.datasets_params.file_ext = 'jpg'; % image data type
%default_params.datasets_params.file_ext = 'png'; % image data type
%default_params.datasets_params.results_folder = 'results_maritime';
%default_params.datasets_params.results_folder = 'results_VOC_5';
default_params.datasets_params.results_folder = 'results_maritime_higher_dimension_v1';
%default_params.datasets_params.results_folder = 'results_VOC_higher_dimension_v1';
%default_params.datasets_params.results_folder = 'results_maritime_using_linsvm_for_esvm_wo_w1';
%default_params.datasets_params.results_folder = 'results_4_categories';
default_params.datasets_params.rename = false;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Features parameters %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
default_params.features_params.cnn_params.feat_name = 'cnn'; 
default_params.features_params.cnn_params.model_folder = 'pretrained_cnn';
default_params.features_params.cnn_params.model_name = 'imagenet-vgg-m.mat';
default_params.features_params.cnn_params.layer = 'relu6'; 
%default_params.features_params.cnn_params.layer = 'prob'; 

default_params.features_params.hog_params.feat_name = 'hog'; 
default_params.features_params.hog_params.sbin = 8;
default_params.features_params.hog_params.width = 150;
default_params.features_params.hog_params.height = 100;
default_params.features_params.hog_extractor = @esvm_extract_hog_feature;
