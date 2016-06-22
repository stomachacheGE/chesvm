%function esvm_show_calibration(new_models, train_datas, neg_set, cls_idx, idx, feat_name, hard_negative, params)

classifi_res_dir = fullfile('.', params.datasets_params.results_folder,'classifications');
esvm_res_root_dir = fullfile(classifi_res_dir, 'esvm');

if hard_negative
    esvm_res_cal_dir = fullfile(esvm_res_root_dir, 'hard_negative', 'calibration');
else
    esvm_res_cal_dir = fullfile(esvm_res_root_dir, 'wo_hard_negative', 'calibration');
end

esvm_res_dir = fullfile(esvm_res_cal_dir, feat_name);

filer = sprintf('%s/%s_esvm_calibration_matrix.mat',esvm_res_dir, feat_name);

if exist(filer,'file')
            i = cls_idx;
            j = idx;
            m = load(new_models{i}{j});
            m = m.m;
            %{
            cal_feature_filers = cal_sets{i}{j}.feat_filers;
            %neg_img_filers = train_datas{qq}.img_filers;
            cal_features = cell(1,length(cal_feature_filers));
                        for filer_i = 1:length(cal_feature_filers)
                 temp = load(cal_feature_filers{filer_i});
                 cal_features{filer_i} = temp.data.feature;
            end
            cal_features = [vertcat(cal_features{:})];
             
            
            num_neg = int16(cal_sets{i}{j}.num_neg);
            cal_features = cal_features(1:size(cal_features,1)-num_neg,:);
            %}
        
        cal_pos_features = cellfun(@(x)x.feature, train_datas{i}, 'UniformOutput', false);
        cal_pos_features = [vertcat(cal_pos_features{:})];
        
        num_pos = size(cal_pos_features,1);
        
        myRandomize;
        ordering = randperm(length(neg_set{i}.feat_filers));
        
        cal_neg_feat_filers = neg_set{i}.feat_filers(ordering(1:num_pos));
        cal_neg_features = cell(num_pos,1);
        for c = 1:num_pos-1
            temp = load(cal_neg_feat_filers{c});
            cal_neg_features{c} = temp.data.feature;
        end
        cal_neg_features = [vertcat(cal_neg_features{:})];
        

            [~, pos_scores] = predict(m.svm_model, cal_pos_features);
            pos_scores = pos_scores(:,2)';

            [pos_scores, indexes] = sort(pos_scores, 'descend');
            %exclude the biggest score, which is the exemplar itself
            indexes = indexes(1:int16(length(pos_scores)/5));
            pos_scores = pos_scores(1:int16(length(pos_scores)/5));
            
            pos_prob = 1 - 0.5 * abs(pos_scores(1) - pos_scores) / (pos_scores(1) - pos_scores(end));
            
            
            [~, neg_scores] = predict(m.svm_model, cal_neg_features);
            neg_scores = neg_scores(:,2)';

            [neg_scores, neg_indexes] = sort(neg_scores, 'descend');
            %exclude the biggest score, which is the exemplar itself
            neg_indexes = neg_indexes(1:int16(length(neg_scores)/5));
            neg_scores = neg_scores(1:int16(length(neg_scores)/5));
            
            neg_prob = 0.5 - 0.5 * abs(neg_scores(1) - neg_scores) / (neg_scores(1) - neg_scores(end));
            
            scores = horzcat(pos_scores, neg_scores);
            prob = horzcat(pos_prob, neg_prob);
            %{
            [~, scores] = predict(m.svm_model, cal_features);
            scores = scores(:,2)';

            [~, indexes] = sort(scores, 'descend');
            
            num_neg = int16(train_datas{i}{j}.num_neg);
            same_cls_scores = scores(indexes(1:int16(num_neg)));
            not_same_cls_scores = scores(end:-1:length(scores)-num_neg+1);
            scores = cat(2, same_cls_scores, not_same_cls_scores);
            ground_truth = [ones(1,length(same_cls_scores)) zeros(1,length(not_same_cls_scores))];
            %}
            fit_sigmoid(scores, prob);
    
else
    fprintf('calibration matrix not found');
end
%
