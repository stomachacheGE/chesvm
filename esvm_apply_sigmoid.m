function prediction = esvm_apply_sigmoid(cal_matrix, test_datas, feat_name, hard_negative, params)


classifi_res_dir = fullfile('.', params.datasets_params.results_folder,'classifications');
esvm_res_root_dir = fullfile(classifi_res_dir, 'esvm');

if hard_negative
    esvm_res_cal_dir = fullfile(esvm_res_root_dir, 'hard_negative', 'calibration');
    esvm_res_dir_wo_cal = fullfile(esvm_res_root_dir, 'hard_negative', 'wo_calibration');
else
    esvm_res_cal_dir = fullfile(esvm_res_root_dir, 'wo_hard_negative', 'calibration');
    esvm_res_dir_wo_cal = fullfile(esvm_res_root_dir, 'wo_hard_negative', 'wo_calibration');
end

if ~exist(esvm_res_cal_dir)
    mkdir(esvm_res_cal_dir)
end

esvm_res_dir_wo_cal = fullfile(esvm_res_dir_wo_cal, feat_name);
esvm_res_dir = fullfile(esvm_res_cal_dir, feat_name); 

if ~exist(esvm_res_dir)
    mkdir(esvm_res_dir)
end

filer = sprintf('%s/%s_esvm_calibration_matrix.mat',esvm_res_dir, feat_name);

if ~exist(filer,'file')
    fprintf(1,'Cannot find calibration matrix for feature %s \n', feat_name);
    return
end

%get number of test images
num_test_images = 0;
for i = 1:length(test_datas)
    num_test_images = num_test_images + numel(test_datas{i});
end

counter = 0;

for i = 1:length(test_datas)
  
  cls_res_dir = fullfile(esvm_res_dir, test_datas{i}{1}.cls_name);
  cls_res_dir_wo_cal = fullfile(esvm_res_dir_wo_cal, test_datas{i}{1}.cls_name);
  
  if ~exist(cls_res_dir)
      mkdir(cls_res_dir)
  end

  for j = 1:length(test_datas{i})
      

      filer = sprintf('%s/%s_%s_score.mat',cls_res_dir_wo_cal, feat_name, test_datas{i}{j}.img_id);
      filer_1 = sprintf('%s/%s_%s_score_sigmoid.mat',cls_res_dir, feat_name, test_datas{i}{j}.img_id);
      
      if ~exist(filer_1, 'file')
          
          result_1 = load(filer);
          scores = result_1.result.scores;
          Index_J_temp = zeros(1, length(scores));
          res = zeros(1, length(scores));
          %outliers = cell(1,length(scores));
          for m = 1:length(scores)
             %res_per_class = zeros(1,length(scores{m}));

             res_per_class = 1 ./ (1 + exp(- cal_matrix{m}(:,1) .* scores{m} + cal_matrix{m}(:,2)));

             [~, Index_J_temp(m)] = max(res_per_class);
             sorted_res_per_class = sort(res_per_class, 'descend');
             %num_exemplar_per_class = size(cal_matrix{i},1);
             %num_to_count = int16(num_exemplar_per_class / 40);
             %res(m) = sum(sorted_res_per_class(1:num_to_count)) / num_to_count;
             %res(m) = sum(sorted_res_per_class(1:10)) / 10;
             %detect outlier, then find which class has bigger outlier mean and
             %regard it as the correct class
             [~, ~, outliers{m}] = deleteoutliers(res_per_class, 0.0001);
             res(m) = mean(res_per_class);
             %[sorted,~] = sort(res_per_class);
             %res(m) = mean(sorted(1:floor(length(res_per_class)/8)));
             %res(m) = mean(sorted(1:10));
             temp{m} = res_per_class;          
          end
          num_outliers = cellfun(@(x) mean(x), outliers, 'UniformOutput', false);
          num_outliers = [horzcat(num_outliers{:})];
          [~,max_idx] = max(num_outliers);
          res(max_idx) = max(temp{max_idx});
          %normalize sum of scores to 1
          %res_max = max(res);
          pos_score_idx = find(res>0);
          [~, Index_I] = max(res);
          if ~isempty(pos_score_idx)
              pos_scores = res(pos_score_idx);
              res = res/sum(pos_scores);
              neg_score_idx = find(res<0);
              res(neg_score_idx) = 0;
          else
              res = -res;
              res = ones(1,size(res,1)) ./ res;
              res = res/sum(res);
          end 
          result.res = res;
          result.i = Index_I;
          result.j = Index_J_temp(Index_I);
          result.scores = temp;
          save(filer_1,'result');
      else 
          file = load(filer_1);
          result = file.result;
      end    
      test_datas{i}{j}.score = result.res;
      test_datas{i}{j}.index_i = result.i;
      test_datas{i}{j}.index_j = result.j;
      
      counter = counter + 1;

      if mod(counter,100) == 0
         fprintf(1,'Apply sigmoid calibration on image %d/%d \n', counter,num_test_images);
      end
  end
end

scores = cell(size(test_datas));

for i = 1:length(test_datas)
    scores_per_test_class = cellfun(@(x)x.score,test_datas{i},'UniformOutput',false);
    scores{i} = [vertcat(scores_per_test_class{:})];
end

scores = [vertcat(scores{:})];

%normalize the sum of scores to 1
%sum_scores_per_test_data = sum(scores,2);
%sum_scores_per_test_data = repmat(sum_scores_per_test_data,1,size(scores,2));

%scores = scores./sum_scores_per_test_data;
[~, indexes] = max(scores,[],2);

prediction.ids = indexes;
prediction.prob = scores;

end