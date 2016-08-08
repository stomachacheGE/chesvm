function linSVMmodel = esvm_train_svm(datas, feat_name, params)
% Train linear SVM.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

models_dir = fullfile('.', params.datasets_params.results_folder,'models');

if ~exist(models_dir, 'dir')
    mkdir(models_dir);
end

model_filer = fullfile(models_dir, sprintf('%s-lin-SVM.mat', feat_name));

if ~exist(model_filer,'file')
    
    datas = [horzcat(datas{:})];
    % Reformat training data so that libsvm can recognize
    features_cell = cellfun(@(x) x.feature, datas, 'UniformOutput', false);
    features = cat(1, features_cell{:});
    features = double(features);
    
    labels_cell = cellfun(@(x) x.label, datas, 'UniformOutput', false);
    labels = cat(1, labels_cell{:});
    
    disp('********* Start train linear SVM  *********');
    feat_size = size(features_cell{1});
    fprintf('Use %s feature with dimension: [%d %d] \n', feat_name, feat_size(1), feat_size(2));

     C_vals = 2.^[-6 -5 -3 -2 -1 0 2 4 6];
    clear i;
    for i=1:length(C_vals);
        % Cross-validation with 5-fold (note the option -v 5)
        opt_string=['-t 0  -v 5 -c ' num2str(2^C_vals(i))];
        xval_acc(i)=svmtrain(labels, features,opt_string);
        fprintf('Parameter selection iteraiton %d with regularization C=%f , accuracy=%f\n', i, C_vals(i),xval_acc(i));
    end

    % select the best C among
    [~,ind]=max(xval_acc);

    % Train the model with the feature vectors
    fprintf('Choose regularization C=%f, strat training... \n',C_vals(ind));
    linSVMmodel = svmtrain(labels,features,['-b 1 -t 0 -c ' num2str(C_vals(ind))]);
    fprintf('Train linear SVM model succeeds. \n');
    save(model_filer,'linSVMmodel');
else
    model = load(model_filer, 'linSVMmodel');
    linSVMmodel = model.linSVMmodel;
    
end
end