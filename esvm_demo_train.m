clear;

addpath(genpath(pwd));

params = esvm_get_default_params;

datasets_info = esvm_get_datasets_info(params.datasets_params);

%use_feature = 'cnn';
use_feature = 'hog';
use_algorithm = 'esvm';
calibration = true;
hard_negative = true;

[train_datas, test_datas] = esvm_initialize_features(datasets_info, ...
                                                     use_feature,use_algorithm,params);

if strcmp(use_algorithm,'svm')
    linearSVMmodel = esvm_train_svm(train_datas, use_feature, params);
    prediction = esvm_predict_svm(linearSVMmodel, test_datas);
else
    [models, cal_set, neg_set] = esvm_train_initialization(train_datas, use_feature);
    
    if hard_negative
        new_models = esvm_train_exemplars_hn(models, neg_set, use_feature, params);
    else
        new_models = esvm_train_exemplars(models, neg_set, use_feature, params);
    end
    
    prediction = esvm_predict(new_models,test_datas, use_feature, hard_negative, params);
    
    if calibration 
        cal_matrix = esvm_perform_calibration(new_models, train_datas, cal_set, use_feature, hard_negative, params);
        prediction = esvm_apply_sigmoid(cal_matrix, test_datas, use_feature, hard_negative, params);
    end
end

ap_res = esvm_evaluate_AP(prediction, test_datas, use_algorithm, ...
                          use_feature, calibration, hard_negative, params);

for i = 1:length(ap_res)
    fprintf(1, 'Class %s has an average precision of %f \n', ...
                upper(ap_res{i}.cls_name), ap_res{i}.ap);
end

