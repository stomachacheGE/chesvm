function esvm_show_calibration(models, cal_sets, cls_idx, idx, feat_name, params)

classifi_res_dir = fullfile('.', params.datasets_params.results_folder,'classifications');
esvm_res_dir = fullfile(classifi_res_dir, 'esvm');

filer = sprintf('%s/%s_esvm_calibration_matrix.mat',esvm_res_dir, feat_name);

if exist(filer,'file')
            i = cls_idx;
            j = idx;
            m = load(models{i}{j});
            m = m.m;
            %{
            cal_feature_filers = cal_sets{i}{j}.feat_filers;
            %neg_img_filers = train_set{qq}.img_filers;
            cal_features = cell(1,length(cal_feature_filers));
                        for filer_i = 1:length(cal_feature_filers)
                 temp = load(cal_feature_filers{filer_i});
                 cal_features{filer_i} = temp.data.feature;
            end
            cal_features = [vertcat(cal_features{:})];
             
            
            num_neg = int16(cal_sets{i}{j}.num_neg);
            cal_features = cal_features(1:size(cal_features,1)-num_neg,:);
            %}
            cal_features = cellfun(@(x)x.feature, cal_sets{i}, 'UniformOutput', false);
            cal_features = [vertcat(cal_features{:})];
            

            [~, scores] = predict(m.svm_model, cal_features);
            scores = scores(:,2)';

            [scores, indexes] = sort(scores, 'descend');
            %exclude the biggest score, which is the exemplar itself
            indexes = indexes(2:int16(length(scores)/5));
            scores = scores(2:int16(length(scores)/5));
            
            ground_truth = 1 - abs(scores(1) - scores) / (scores(1) - scores(end));
            
        fit_sigmoid(scores, ground_truth);
    
    
else
    fprintf('calibration matrix not found');
end
end
