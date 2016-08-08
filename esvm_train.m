function esvm_train(feature, algorithm, calibration, hard_negative)
% This is the main interface of the package. Make sure all parameters in
% esvm_get_default_params are set correctly before you run this function.
%
% [feature]: String, supports 'cnn', 'hog', 'cnnhog'.
% [algorithm]: String, supports 'svm', 'esvm'.
% [calibration]: Boolean, decide whether calibration should be used.  
% [hard_negative]: Boolean, decide whether hard negative mining should be
%                  used for trainig exemplar-SVMs.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

params = esvm_get_default_params;

datasets_info = esvm_get_datasets_info(params.datasets_params);

use_feature = feature;
use_algorithm = algorithm;

[train_datas, test_datas] = esvm_initialize_features(datasets_info, ...
                                                     use_feature,use_algorithm,params);
if strcmp(use_algorithm,'svm')
    % train a linear SVM
    linearSVMmodel = esvm_train_svm(train_datas, use_feature, params);
    prediction = esvm_predict_svm(linearSVMmodel, test_datas);
else
    % initialize training models and get calibration set
    [models, cal_set, neg_set] = esvm_train_initialization(train_datas, use_feature);
    
    if hard_negative
        new_models = esvm_train_exemplars_hn(models, neg_set, cal_set, use_feature, params);
    else
        new_models = esvm_train_exemplars(models, neg_set, cal_set, use_feature, params);
    end
    % make predcition based on test data and the trained models
    prediction = esvm_predict(new_models,test_datas, use_feature, hard_negative, params);
    
    if calibration 
        % if calibration enables , fit sigmoid functions and put parameters learned into a matrix
        cal_matrix = esvm_perform_calibration(new_models, train_datas, use_feature, hard_negative, params);
        % apply calibration on exsiting prediciton scores
        prediction = esvm_apply_sigmoid(cal_matrix, test_datas, use_feature, hard_negative, params);
    end
end
% evaluate predictions , calculate accuracies and display confusion matrix
res = esvm_evaluate_ACC(prediction, test_datas, use_algorithm, ...
                          use_feature, calibration, hard_negative, params);

for i = 1:length(res.classes)
    fprintf(1, 'Class %s has an accuraccy of %f \n', ...
                upper(res.classes{i}), res.acc_per_cls(i));
end

end
