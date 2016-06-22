function cal_mat = esvm_perform_calibration(models, train_set, cal_set, feat_name, hard_negative, params)

classifi_res_dir = fullfile('.', params.datasets_params.results_folder,'classifications');
esvm_res_root_dir = fullfile(classifi_res_dir, 'esvm');

if hard_negative
    esvm_res_cal_dir = fullfile(esvm_res_root_dir, 'hard_negative', 'calibration');
else
    esvm_res_cal_dir = fullfile(esvm_res_root_dir, 'wo_hard_negative', 'calibration');
end

esvm_res_dir = fullfile(esvm_res_cal_dir, feat_name);

filer = sprintf('%s/%s_esvm_calibration_matrix.mat',esvm_res_dir, feat_name);

if ~exist(esvm_res_dir, 'dir')
    mkdir(esvm_res_dir)
end


if ~exist(filer,'file')

    num_models = 0;
    for i = 1:length(models)
        num_models = num_models + numel(models{i});
    end

    counter = 1;
    cal_mat = cell(1, length(models));
    for i=1:length(models)

        cls_cal_mat = cell(1,length(models{i}));
        cal_pos_features = cellfun(@(x)x.feature, train_set{i}, 'UniformOutput', false);
        cal_pos_features = [vertcat(cal_pos_features{:})];
        

        
        for j=1:length(models{i})

            m = load(models{i}{j});
            m = m.m;
            %{
            cal_feature_filers = train_set{i}{j}.feat_filers;
            %neg_img_filers = train_set{qq}.img_filers;
            cal_features = cell(1,length(cal_feature_filers));

            for filer_i = 1:length(cal_feature_filers)
                 temp = load(cal_feature_filers{filer_i});
                 cal_features{filer_i} = temp.data.feature;
            end
           
            cal_features = [vertcat(cal_features{:})]; 
            
            num_neg = int16(train_set{i}{j}.num_neg);
            cal_features = cal_features(1:size(cal_features,1)-num_neg,:);
             %}
            [~, all_pos_scores] = predict(m.svm_model, cal_pos_features);
            all_pos_scores = all_pos_scores(:,2)';

            [sorted_pos_scores, indexes] = sort(all_pos_scores, 'descend');
            %exclude the biggest score, which is the exemplar itself
            
            num_pos = length(sorted_pos_scores);
            num_neg = length(cal_set{i}{j}.neg_filer);
            idx = int16(1:num_neg);
            idxs = int16(floor(num_pos / num_neg)) * idx;           
            pos_scores = sorted_pos_scores(idxs);
            if strcmp(feat_name,'cnn')
            pos_prob = 1 - 0.5 * abs(pos_scores(1) - pos_scores) / (pos_scores(1) - pos_scores(end));
            

 
            cal_neg_feat_filers = cal_set{i}{j}.neg_filer;
            cal_neg_features = cell(num_neg,1);
            for c = 1:num_neg
                temp = load(cal_neg_feat_filers{c});
                cal_neg_features{c} = temp.data.feature;
            end
            cal_neg_features = [vertcat(cal_neg_features{:})];

            [~, neg_scores] = predict(m.svm_model, cal_neg_features);
            neg_scores = neg_scores(:,2)';

            [neg_scores, neg_indexes] = sort(neg_scores, 'ascend');
            
            neg_prob = 0.3 * (1 - 1 * abs(neg_scores(end) - neg_scores) / (neg_scores(end) - neg_scores(1)));
            
            scores = horzcat(pos_scores, neg_scores);
            prob = horzcat(pos_prob, neg_prob);
            else
                
            
            pos_prob = 1 - abs(pos_scores(1) - pos_scores) / (pos_scores(1) - pos_scores(end));
            prob = pos_prob;
            scores = pos_scores;
            end
            
            %{
            [~, scores] = predict(m.svm_model, cal_features);
            scores = scores(:,2)';

            [~, indexes] = sort(scores, 'descend');
            
            num_neg = int16(train_set{i}{j}.num_neg);
            same_cls_scores = scores(indexes(1:int16(num_neg)));
            not_same_cls_scores = scores(end:-1:length(scores)-num_neg+1);
            scores = cat(2, same_cls_scores, not_same_cls_scores);
            ground_truth = [ones(1,length(same_cls_scores)) zeros(1,length(not_same_cls_scores))];
            %}
            m.sigmoid_coef = fit_sigmoid(scores, prob);
            cls_cal_mat{j} = m.sigmoid_coef; 
            counter = counter + 1;


          if mod(counter,50) == 0
          fprintf(1,'Getting sigmoid coefficients %d/%d \n', counter, num_models);
          end
        end
        cal_mat{i} = [vertcat(cls_cal_mat{:})];
    end
    
    save(filer, 'cal_mat');
else
    cal_mat = load(filer);
    cal_mat = cal_mat.cal_mat;
end
end
