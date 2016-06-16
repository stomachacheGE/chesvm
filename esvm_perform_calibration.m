function cal_mat = esvm_perform_calibration(models, cal_sets, feat_name, hard_negative, params)

classifi_res_dir = fullfile('.', params.datasets_params.results_folder,'classifications');
esvm_res_dir = fullfile(classifi_res_dir, 'esvm');

if ~exist(classifi_res_dir,'dir')
    mkdir(classifi_res_dir);
end

if ~exist(esvm_res_dir,'dir')
    mkdir(esvm_res_dir);
end

if hard_negative
    filer = sprintf('%s/%s_esvm_calibration_matrix.mat',esvm_res_dir, feat_name);
else
    filer = sprintf('%s/%s_esvm_calibration_matrix_wo_hn.mat',esvm_res_dir, feat_name);
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
        cal_features = cellfun(@(x)x.feature, cal_sets{i}, 'UniformOutput', false);
        cal_features = [vertcat(cal_features{:})];
        for j=1:length(models{i})

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
            [~, scores] = predict(m.svm_model, cal_features);
            scores = scores(:,2)';

            [scores, indexes] = sort(scores, 'descend');
            %exclude the biggest score, which is the exemplar itself
            indexes = indexes(2:int16(length(scores)/5));
            scores = scores(2:int16(length(scores)/5));
            
            ground_truth = 1 - abs(scores(1) - scores) / (scores(1) - scores(end));
            
            %{
            [~, scores] = predict(m.svm_model, cal_features);
            scores = scores(:,2)';

            [~, indexes] = sort(scores, 'descend');
            
            num_neg = int16(cal_sets{i}{j}.num_neg);
            same_cls_scores = scores(indexes(1:int16(num_neg)));
            not_same_cls_scores = scores(end:-1:length(scores)-num_neg+1);
            scores = cat(2, same_cls_scores, not_same_cls_scores);
            ground_truth = [ones(1,length(same_cls_scores)) zeros(1,length(not_same_cls_scores))];
            %}
            m.sigmoid_coef = fit_sigmoid(scores, ground_truth);
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
