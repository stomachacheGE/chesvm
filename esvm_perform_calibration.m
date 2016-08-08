function cal_mat = esvm_perform_calibration(models, train_set, feat_name, hard_negative, params)
% fit sigmoid function for each exemplar-SVM and put all learned parameters into a matrix
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

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
            [neg_scores, neg_indexes] = sort(neg_scores, 'ascend');
     
            % delete the positive scores which are smaller than
            % the maximum of the negative scores
            [~, del_idxs] = find(sorted_pos_scores<max(neg_scores));
            pos_scores = sorted_pos_scores;
            pos_scores(del_idxs) = [];

            % recover 2 positive scores if the above step returns none or
            % only one positive score
            if numel(pos_scores) == 0 || numel(pos_scores) == 1
                pos_scores = sorted_pos_scores(1:2);
                neg_scores = neg_scores(1:end-2);
            end
            
            % assume probabilites of positive scores are distributed from 0.5 to 1,
            % and those of negative score from 0 to 0.3
            pos_prob = 1 - 0.5 * abs(pos_scores(1) - pos_scores) / (pos_scores(1) - pos_scores(end));
            neg_prob = 0.3 * (1 - 1 * abs(neg_scores(end) - neg_scores) / (neg_scores(end) - neg_scores(1)));
            
            scores = horzcat(pos_scores, neg_scores);
            prob = horzcat(pos_prob, neg_prob);
            
            % fit sigmoid function given scores and probabilities
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
