% -------------------------------------------------------------------------
% 
%              Demonstrator Script for Image Classification Tasks
% 
%
% The example should illustrate the usage of pretrained convolutional 
% neural network (CNN) model as automatic feature extractor in combination
% with SVM classifier
%
%
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% File handling is based on ICT'13 Bag-of-Words Workshop, Instructors: L. Ballan / L. Seidenari                                  
% -------------------------------------------------------------------------            
% -------------------------------------------------------------------------
% Date:     11.04.16
% Modified: Gediminas Simkus
% remarks:  < tested >
% -------------------------------------------------------------------------

clear;

% get and add current path
%fp = fileparts(which(mfilename));
%addpath(genpath(fp));
%cd(fp);
%format short; % output short

params = esvm_get_default_params;

datasets = esvm_get_datasets_info(params.datasets_params);



% Pretrained CNN Model PATH
% Download models if needed from http://www.vlfeat.org/matconvnet/pretrained/
cnn_dir = fullfile(pwd, 'pretrained_cnn');
% cnn_dir = '/media/datadrive/Various/CNN/pretrained_cnn/';
cnn_model_name = 'imagenet-vgg-m.mat';
cnn_path = fullfile(cnn_dir,cnn_model_name); 
if ~exist(cnn_path,'file')
    modelURL = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat';
    fprintf('Downloading CNN model... may take a while... ');
    websave(cnn_path,modelURL);  
    fprintf('Done.\n');
end

% FRAMEWORK PATHS
% LibSVM
basepath = '..';
libsvmpath = fullfile(pwd, '..', 'lib','libsvm-3.11','matlab'); % path to the SVM lib
addpath(libsvmpath);
dataset_path = fullfile(basepath,imgFolder,dataset_dir); 
% MatConvNet
mcnpath = fullfile(pwd, '..', 'lib','matconvnet-1.0-beta18','matlab');
run(fullfile(mcnpath,'vl_setupnn.m'));

if strcmp(desc_name,'cnn')
    disp('**** CNN with SVM ****')
% elseif OTHER METHOD           
end
%% Create a new dataset split / or load previous one
if do_create_predefined_sets    
    data = create_set_structure(fullfile(basepath, imgFolder, ...
                    dataset_dir),num_train_img,num_test_img,file_ext, out_cnn_name);
    save(fullfile(dataset_path,file_split),'data');
else
    load(fullfile(dataset_path,file_split));
end
classes = {data.classname}; % create cell array of class name strings

%% Compute Multi-Scale Dense SIFT (MSDSIFT) or CNN features for every image in each class
if do_feat_extraction
    % note that the feature extraction will be done for all images inside
    % the data set
    if strcmp(desc_name,'cnn')
        disp('*** CNN Feature extraction ***');
    % elseif
        % OTHER METHOD
    end    
    extract_features(fullfile('..',imgFolder,dataset_dir), desc_name, file_ext, '/train', cnn_path, out_cnn_name);
    extract_features(fullfile('..',imgFolder,dataset_dir), desc_name, file_ext, '/test', cnn_path, out_cnn_name);
end

%% Load pre-computed Features

% Load precomputed features for training images
lasti=1;
for i = 1:length(data)
    fprintf('Loading precomputed features for training class %s \n',upper(data(i).classname));
    images_descs = get_feature_files(data,i,file_ext,desc_name,'train');
    for j = 1:length(images_descs)
        fname = fullfile(dataset_path,data(i).classname,images_descs{j});
        tmp = load(fname,'-mat');
        tmp.desc.class=i;      
        fname=regexprep(fname,['.' desc_name],['.' file_ext]);
        tmp.desc.imgfname=fullfile(pwd,regexprep(fname,['/feats_' out_cnn_name],''));             
        desc_train(lasti)=tmp.desc; 
        desc_train(lasti).cnnfeature = single(desc_train(lasti).cnnfeature);
        lasti=lasti+1;
    end
end

% Descriptors of test images 
lasti=1;
for i = 1:length(data)
    fprintf('Loading descriptors for test class %s \n',upper(data(i).classname));
    images_descs = get_feature_files(data,i,file_ext,desc_name,'test');
    for j = 1:length(images_descs)
        fname = fullfile(dataset_path,data(i).classname,images_descs{j});
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        fname=regexprep(fname,['.' desc_name],['.' file_ext]);
        tmp.desc.imgfname=fullfile(pwd,regexprep(fname,'/feats',''));
        desc_test(lasti)=tmp.desc; 
        desc_test(lasti).cnnfeature = single(desc_test(lasti).cnnfeature);
        lasti=lasti+1;
    end
end

%% PRINT PARAMS
fprintf('** PARAMETERS **\ndataset_path = %s\n',dataset_path);
fprintf('num_test_img = %i, num_train_img = %i\n',num_test_img,num_train_img);
fprintf('desc_name = %s\n',desc_name);
if strcmp(desc_name,'cnn')
    fprintf('cnn_path = %s \n',cnn_path);
    fprintf('cnn_path = %s \n',out_cnn_name);
end
fprintf('----------------------------------------------\n');
%% SVM learning and classification 
% Train a classifier using extracted features and calculate accuracy

% Prepare data for classification
% Concatenate faetures into training and test matrices 
if strcmp(desc_name,'cnn')
    feat_train = cat(1,desc_train.cnnfeature); feat_train = double(feat_train);
    feat_test = cat(1,desc_test.cnnfeature); feat_test = double(feat_test);
% else
    % other feats
end

% Concatenate labels
labels_train=cat(1,desc_train.class);
labels_test=cat(1,desc_test.class);

disp('*** Start linear SVM - model creation and classification ***');
% Cross-validation with 5-fold (note the obtion -v 5)
C_vals=log2space(7,10,5);
for i=1:length(C_vals);
    opt_string=['-t 0  -v 5 -c ' num2str(C_vals(i))];
    xval_acc(i)=svmtrain(labels_train,feat_train,opt_string);
end
% select the best C among
[~,ind]=max(xval_acc);

% Train the model with the feature vectors
linSVMmodel = svmtrain(labels_train,feat_train,['-b 1 -t 0 -c ' num2str(C_vals(ind))]);

    
disp('*** Linear SVM test label prediction ***');
% Predict class label id's with the probability estimates
[predicted_ids,~,prob_estim] = svmpredict(labels_test,feat_test,linSVMmodel, '-b 1');   

% Classification confusion matrix and classification accuracy 
CM = confusionmat(labels_test,predicted_ids);
CM = CM./repmat(sum(CM,2), [1 size(CM,2)]); % normalize confusion matrix for each class
oacc_lin = sum(labels_test==predicted_ids)/numel(labels_test); % total classification accuracy
macc_lin = mean(diag(CM)); % mean per-class classification accuracy
fprintf('\nOVERALL ACCURACY Linear-SVM classification: %1.4f\n',oacc_lin );
fprintf('MEAN PER-CLASS ACCURACY Linear-SVM  classification: %1.4f\n',macc_lin);
disp('classes:'), disp(classes');
disp('Classification confusion matrix:'), disp(CM);

%% Save misclassed images to folder
savemisclass(labels_test, predicted_ids, desc_test, classes, 'linSVM');

