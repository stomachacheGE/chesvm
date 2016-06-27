function esvm_train(feature, calibration,algorithm, hard_negative)

% get and add current path
%fp = fileparts(which(mfilename));
%addpath(genpath(fp));
%cd(fp);
%format short; % output short


params = esvm_get_default_params;

datasets_info = esvm_get_datasets_info(params.datasets_params);

use_feature = feature;
%use_feature = 'hog';
use_algorithm = algorithm;
calibration = calibration;
hard_negative = hard_negative;

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

res = esvm_evaluate_ACC(prediction, test_datas, use_algorithm, ...
                          use_feature, calibration, hard_negative, params);

for i = 1:length(res.classes)
    fprintf(1, 'Class %s has an accuraccy of %f \n', ...
                upper(res.classes{i}), res.acc_per_cls(i));
end



end
