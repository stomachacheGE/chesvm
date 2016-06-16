function linSVMmodel = esvm_train_svm(datas, feat_name, params)

models_dir = fullfile('.', params.datasets_params.results_folder,'models');


if ~exist(models_dir, 'dir')
    mkdir(models_dir);
end

model_filer = fullfile(models_dir, sprintf('%s-lin-SVM.mat', feat_name));

if ~exist(model_filer,'file')
    %{
    num_class = length(feature_files);
    num_image = 0;
    for n=1:num_class
        num_image = num_image + length(feature_files{n}.train_features);
    end
    
    
    features_cell = cell(num_image,1);
    labels = zeros(num_image,1);

    counter = 1;

    for i=1:num_class
        num_feature = length(feature_files{i}.train_features);
        for j=1:num_feature
            feature = load(feature_files{i}.train_features{j});
            features_cell{counter} = feature;
            labels(counter) = i;
            counter = counter +1;
        end

    end

    len = length(feature.feature);
    features = zeros(num_image, len);
    for i = 1:num_image
        features(i,:) = features_cell{i}.feature;
    end
    %}

    datas = [horzcat(datas{:})];
    
    features_cell = cellfun(@(x) x.feature, datas, 'UniformOutput', false);
    features = cat(1, features_cell{:});
    features = double(features);
    
    labels_cell = cellfun(@(x) x.label, datas, 'UniformOutput', false);
    labels = cat(1, labels_cell{:});
    %labels = double(labels);
    
    disp('********* Start train linear SVM  *********');
    feat_size = size(features_cell{1});
    fprintf('Use %s feature with dimension: [%d %d] \n', feat_name, feat_size(1), feat_size(2));
    % Cross-validation with 5-fold (note the obtion -v 5)
    %C_vals=log2space(7,10,5);
     C_vals = 2.^[-6 -5 -3 -2 -1 0 2 4 6];
    clear i;
    for i=1:length(C_vals);
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