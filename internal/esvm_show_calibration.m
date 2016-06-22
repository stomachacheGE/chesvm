%function esvm_show_calibration(new_models, train_datas, neg_set, cls_idx, idx, feat_name, hard_negative, params)


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
        
        cls_cal_mat = cell(1,length(models{i}));
        cal_pos_features = cellfun(@(x)x.feature, train_datas{i}, 'UniformOutput', false);
        cal_pos_features = [vertcat(cal_pos_features{:})];
        
        num_neg = length(cal_set{i}{j}.neg_filer);

        cal_neg_feat_filers = cal_set{i}{j}.neg_filer;
        cal_neg_features = cell(num_neg,1);
        for c = 1:num_neg
            temp = load(cal_neg_feat_filers{c});
            cal_neg_features{c} = temp.data.feature;
        end
        cal_neg_features = [vertcat(cal_neg_features{:})];

            %{
            cal_feature_filers = train_datas{i}{j}.feat_filers;
            %neg_img_filers = train_datas{qq}.img_filers;
            cal_features = cell(1,length(cal_feature_filers));

            for filer_i = 1:length(cal_feature_filers)
                 temp = load(cal_feature_filers{filer_i});
                 cal_features{filer_i} = temp.data.feature;
            end
           
            cal_features = [vertcat(cal_features{:})]; 
            
            num_neg = int16(train_datas{i}{j}.num_neg);
            cal_features = cal_features(1:size(cal_features,1)-num_neg,:);
             %}
            [~, all_pos_scores] = predict(m.svm_model, cal_pos_features);
            all_pos_scores = all_pos_scores(:,2)';

            [sorted_pos_scores, indexes] = sort(all_pos_scores, 'descend');
            %exclude the biggest score, which is the exemplar itself
            num_pos = length(sorted_pos_scores);
            idx = int16(1:num_neg);
            idxs = int16(floor(num_pos / num_neg)) * idx;           
            pos_scores = sorted_pos_scores(idxs);
            

            if strcmp(feat_name,'cnn')
            pos_prob = 1 - 0.5 * abs(pos_scores(1) - pos_scores) / (pos_scores(1) - pos_scores(end));
            
            
            [~, neg_scores] = predict(m.svm_model, cal_neg_features);
            neg_scores = neg_scores(:,2)';

            [neg_scores, neg_indexes] = sort(neg_scores, 'ascend');
            
            neg_prob = 0.3 * (1 - 1 * abs(neg_scores(end) - neg_scores) / (neg_scores(end) - neg_scores(1)));
            
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
            else
                
            
            pos_prob = 1 - abs(pos_scores(1) - pos_scores) / (pos_scores(1) - pos_scores(end));
            prob = pos_prob;
            scores = pos_scores;
            end
            m.sigmoid_coef = fit_sigmoid(scores, prob);
    

%
