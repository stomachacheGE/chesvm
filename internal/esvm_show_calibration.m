function esvm_show_calibration( cls_idx, idx, feat_name, new_models, train_datas)
% Show the fitted sigmoid function for an exemplar-SVM which is
% specified by cls_idx and idx.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

i = cls_idx;
j = idx;
m = load(new_models{i}{j});
m = m.m;

cls_cal_mat = cell(1,length(new_models{i}));
cal_pos_features = cellfun(@(x)x.feature, train_datas{i}, 'UniformOutput', false);
cal_pos_features = [vertcat(cal_pos_features{:})];

num_neg = length(m.cal_set);

cal_neg_feat_filers = m.cal_set;
cal_neg_features = cell(num_neg,1);
for c = 1:num_neg
    temp = load(cal_neg_feat_filers{c});
    cal_neg_features{c} = temp.data.feature;
end
cal_neg_features = [vertcat(cal_neg_features{:})];

[~, all_pos_scores] = predict(m.svm_model, cal_pos_features);
all_pos_scores = all_pos_scores(:,2)';
[sorted_pos_scores, indexes] = sort(all_pos_scores, 'descend');


[~, neg_scores] = predict(m.svm_model, cal_neg_features);
neg_scores = neg_scores(:,2)';

[~, del_idxs] = find(sorted_pos_scores<max(neg_scores));
sorted_pos_scores(del_idxs) = [];
pos_scores = sorted_pos_scores;

pos_prob = 1 - 0.5 * abs(pos_scores(1) - pos_scores) / (pos_scores(1) - pos_scores(end));

[neg_scores, neg_indexes] = sort(neg_scores, 'ascend');            
neg_prob = 0.3 * (1 - 1 * abs(neg_scores(end) - neg_scores) / (neg_scores(end) - neg_scores(1)));

scores = horzcat(pos_scores, neg_scores);
prob = horzcat(pos_prob, neg_prob);

m.sigmoid_coef = fit_sigmoid(scores, prob);
end
    


