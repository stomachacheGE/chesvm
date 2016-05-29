%{
clear;

addpath(genpath(pwd));
% get and add current path
%fp = fileparts(which(mfilename));
%addpath(genpath(fp));
%cd(fp);
%format short; % output short

params = esvm_get_default_params;

datasets_info = esvm_get_datasets_info(params.datasets_params);

%use_feature = 'cnn';
use_feature = 'hog';
use_algorithm = 'esvm';

feature_files = cell(1,length(datasets_info));

[train_datas, test_datas] = esvm_initialize_features(datasets_info,use_feature,use_algorithm,params);

%data = esvm_construct_data(train_datas, datasets_info, params);

[models, neg_set] = esvm_train_initialization(train_datas, use_feature);
%}
%% Train Exemplar SVM models 
qq = 1;

pos_features = cellfun(@(x)x.feature, train_datas{1,1}, 'UniformOutput', false);
pos_features = [vertcat(pos_features{:})];

num_pos = size(pos_features,1);

%models = cell(num_pos,1);
extras = cell(num_pos,1);

Mus = cell(num_pos,1);
Sigmas = cell(num_pos,1);
Biases = cell(num_pos,1);
Betas = cell(num_pos,1);

for i = 1:num_pos
        
    filer = sprintf('/home/liangfu/Desktop/models/%d.mat', i);
    
    feat_name = 'cnn';
    neg_feature_filers = neg_set{qq}.feat_filers;
    neg_img_filers = neg_set{qq}.img_filers;
    neg_features = cell(1,length(neg_feature_filers));


    for filer_i = 1:length(neg_feature_filers)
       if strcmp(feat_name,'cnn')
         temp = load(neg_feature_filers{filer_i});
         neg_features{filer_i} = temp.data.feature;
       else
         %{
         img = imread(neg_img_filers{filer_i});
         img = imresize(double(img), m.img_size);
         %fprintf(1,'%d size img is [%d %d] \n',index, size(img,1),size(img,2));
         [temp, ~] = params.training_params.hog_extractor(img);
         neg_features{filer_i} = temp;
         %}
         temp = load(neg_feature_filers{filer_i});
         neg_features{filer_i} = temp.data.feature;
       end
    end
    neg_features = [vertcat(neg_features{:})];     
    pos_feature = pos_features(i,:);
    train_features = vertcat(pos_feature, neg_features);

    neg_labels = ones(length(neg_feature_filers),1);
    neg_labels = -neg_labels;
    train_labels = vertcat(1, neg_labels) ;


    weights = ones(size(train_features,1),1);
    weights(1) = 5000;
        
    if exist(filer,'file')
        model = load(filer);
        model = model.model;
    else


        fprintf(1,' Strat traning %d/%d model \n', i, num_pos);
        %models{i} = fitcsvm(train_features, train_labels, 'KernelFunction', 'linear', 'PolynomialOrder', ...
        %                [], 'KernelScale', 1, 'BoxConstraint', 10^-7, 'Standardize', 1,...
        %                'ClassNames', [-1; 1], 'Weights', weights);
        model = fitcsvm(train_features, train_labels, 'KernelFunction', 'linear', 'PolynomialOrder', ...
                        [], 'KernelScale', 1, 'BoxConstraint', 10^-7, 'Standardize', 1,...
                        'ClassNames', [-1; 1], 'Weights', weights);
        save(filer, 'model');
    end
    [~,extras{i}.score] = predict(model, pos_features);
    %Scores(:, end:-1:1) = score(:,:); % Second column contains positive-class scores
    extras{i}.num_postives = sum(extras{i}.score(:,2)>0);
    
    Mus{i} = model.Mu;
    Sigmas{i} = model.Sigma;
    Biases{i} = model.Bias;
    Betas{i} = model.Beta';
end


%% Exemplar SVM prediction
%Mus = cellfun(@(x)x.Mu,models,'UniformOutput',false);
%Sigmas = cellfun(@(x)x.Sigma,models,'UniformOutput',false);
%Betas = cellfun(@(x)x.Beta',models,'UniformOutput',false);
%Biases = cellfun(@(x)x.Bias,models,'UniformOutput',false);
Mus = [vertcat(Mus{:})];
Sigmas = [vertcat(Sigmas{:})];
Betas = [vertcat(Betas{:})];
Biases = [vertcat(Biases{:})];

test_datas_1 = horzcat(test_datas{:});
tests = cellfun(@(x)x.feature, test_datas_1, 'UniformOutput', false);
tests = [vertcat(tests{:})];
scores = zeros(size(tests,1),1);
predicted = zeros(size(tests,1),1);

for i = 1:length(scores)
    input = tests(i,:);
    standarized_inputs = (repmat(input, size(Mus,1), 1) - Mus ) ./ Sigmas;
    %remove NaN entry
    [row, col] = find(isnan(standarized_inputs));
    standarized_inputs(row,col) = 0;
    scores(i) = max(sum(standarized_inputs  .* Betas, 2) + Biases);
    if scores(i) >= 0
        predicted(i) = 1;
    else
        predicted(i) = -1;
    end
    
end


