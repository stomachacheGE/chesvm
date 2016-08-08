function prediction = esvm_apply_sigmoid(cal_matrix, test_datas, feat_name, hard_negative, params)
% apply the learned sigmoid function to the predictions and return new
% pridiction scores.
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

          for m = 1:length(scores)
             % use the learned sigmoid function to get probabilities
             res_per_class = 1 ./ (1 + exp(- cal_matrix{m}(:,1) .* scores{m} + cal_matrix{m}(:,2)));
             [res(m), Index_J_temp(m)] = max(res_per_class);            
             temp{m} = res_per_class;
          end
          
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
[~, indexes] = max(scores,[],2);

prediction.ids = indexes;
prediction.prob = scores;

end