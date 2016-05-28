function prediction = esvm_predict_svm(model, test_datas )


test_datas = [horzcat(test_datas{:})];

features_cell = cellfun(@(x) x.feature, test_datas, 'UniformOutput', false);
features = cat(1, features_cell{:});
%features = double(features);

labels_cell = cellfun(@(x) x.label, test_datas, 'UniformOutput', false);
labels = cat(1, labels_cell{:});

disp('*** Linear SVM test label predicting... ***');
% Predict class label id's with the probability estimates
[predicted_ids,~,prob_estim] = svmpredict(labels,features,model, '-b 1');   

prediction.ids = predicted_ids;
prediction.prob = prob_estim;
end