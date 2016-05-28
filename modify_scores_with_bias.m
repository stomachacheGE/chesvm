%function prediction = modify_score(models, test_datas, feat_name, params)

feat_name = 'hog';
redecide_prob = true;
redecide_bias = false;

classifi_res_dir = fullfile('.', params.datasets_params.results_folder,'classifications');
esvm_res_dir = fullfile(classifi_res_dir, 'esvm');
models_dir = fullfile('.', params.datasets_params.results_folder,'models');
%get number of test images
num_test_images = 0;
for i = 1:length(test_datas)
    num_test_images = num_test_images + numel(test_datas{i});
end
%get number of models
num_models = 0;
for i = 1:length(models)
    num_models = num_models + numel(models{i});
end

counter = 0;

if redecide_bias
    biases = cell(length(models),1);
    for m = 1:length(models)

        biases{m} = zeros(1,length(models{m}));

     for n = 1:length(models{m}) 

          model = models{m}{n};
          model_filer = sprintf('%s/%s/%s_esvm_%s_%s_wo_hn.mat',models_dir, model.cls_name, feat_name, model.cls_name, model.img_id);
          model_struct = load(model_filer);
          model_struct = model_struct.m;
          biases{m}(1,n) = model_struct.self_bias;
     end   
    end
end

for i = 1:length(test_datas)
  
  cls_res_dir = fullfile(esvm_res_dir, test_datas{i}{1}.cls_name);

  for j = 1:length(test_datas{i})
      
      model_counter = 0;
      
      res = zeros(size(models));
      
      filer = sprintf('%s/%s_%s_score.mat',cls_res_dir, feat_name, test_datas{i}{j}.img_id);
      
      if exist(filer,'file')
         
          result = load(filer);
          result = result.result;
          
          for m = 1:length(models)
            if redecide_bias
                result.scores{m} = result.scores{m} + biases{m};
            end
          end 
          
          if redecide_prob
              
              for m = 1:length(result.scores)

                 res_per_class = result.scores{m};
                 [res(m), Index_J_temp(m)] = max(res_per_class);
                 %[sorted,~] = sort(res_per_class);
                 %res(m) = mean(sorted(1:floor(length(res_per_class)/8)));
                 %res(m) = mean(sorted(1:10));
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
              [~, Index_I_1] = max(res);
              Index_J = Index_J_temp(Index_I);
              
              result.res = res;
              result.i = Index_I;
              result.j = Index_J;
          end
         esvm_res_dir_1 = fullfile(classifi_res_dir, 'esvm_1');
         if ~exist(esvm_res_dir_1, 'dir')
             mkdir(esvm_res_dir_1)
         end
         cls_res_dir_1 = fullfile(esvm_res_dir_1, test_datas{i}{1}.cls_name);
         if ~exist(cls_res_dir_1, 'dir')
             mkdir(cls_res_dir_1)
         end
         filer_1 = sprintf('%s/%s_%s_score.mat',cls_res_dir_1, feat_name, test_datas{i}{j}.img_id);
         save(filer_1,'result');
      else 
         fprintf(1,'Score result from %d does not exist \n', counter);
      end
  end
end

