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

[train_datas, val_datas, test_datas] = esvm_initialize_features(datasets_info,use_feature,use_algorithm,params);

%data = esvm_construct_data(train_datas, datasets_info, params);

if strcmp(use_algorithm,'svm')
    linearSVMmodel = esvm_train_svm(train_datas, val_datas, use_feature, params);
    prediction = esvm_predict_svm(linearSVMmodel, test_datas);
else
    [models, neg_set] = esvm_train_initialization(train_datas, use_feature);
    new_models = esvm_train_exemplars(models, train_datas, neg_set, use_algorithm, use_feature,params);
    val_matrix = esvm_perform_validation(new_models, val_datas);
    esvm_predict(new_models,test_datas, use_feature, params,val);
    prediction = esvm_apply_sigmoid(new_models, test_datas, feat_name, params, val_matrix)

end



ap_res = esvm_evaluate_AP(prediction, test_datas, use_algorithm, use_feature, params);

for i = 1:length(ap_res)
    fprintf(1, 'Class %s has an average precision of %f \n', upper(ap_res{i}.cls_name), ap_res{i}.ap);
end
%model.w * test_datas{1}{59}.feature' - model.b;
