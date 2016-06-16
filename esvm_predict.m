function prediction = esvm_predict(models, test_datas, feat_name, params)



classifi_res_dir = fullfile('.', params.datasets_params.results_folder,'classifications');
esvm_res_dir = fullfile(classifi_res_dir, 'esvm');

if ~exist(classifi_res_dir,'dir')
    mkdir(classifi_res_dir);
end

if ~exist(esvm_res_dir,'dir')
    mkdir(esvm_res_dir);
end

for i = 1:length(test_datas)
  
  cls_res_dir = fullfile(esvm_res_dir, test_datas{i}{1}.cls_name);
  if ~exist(cls_res_dir, 'dir')
        mkdir(cls_res_dir);
  end
end

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

filer_1 = sprintf('%s/%s_models_in_matrix.mat', esvm_res_dir, feat_name);

if ~exist(filer_1,'file')
    fprintf(1,'Extracting models into one file... \n');
    Mus_cell = cell(1,length(models));
    Sigmas_cell = cell(1,length(models));
    Biases_cell = cell(1,length(models));
    Betas_cell = cell(1,length(models));
    
    for i=1:length(models)
        num_per_class = length(models{i});
        Mus = cell(num_per_class,1);
        Sigmas = cell(num_per_class,1);
        Biases = cell(num_per_class,1);
        Betas = cell(num_per_class,1);
        for j=1:length(models{i})
            m = load(models{i}{j});
            model = m.m.svm_model;
            Mus{j} = model.Mu;
            Sigmas{j} = model.Sigma;
            Biases{j} = model.Bias;
            Betas{j} = model.Beta';
        end

        Mus = [vertcat(Mus{:})];
        Sigmas = [vertcat(Sigmas{:})];
        Betas = [vertcat(Betas{:})];
        Biases = [vertcat(Biases{:})];

        Mus_cell{i} = Mus;
        Sigmas_cell{i} = Sigmas;
        Biases_cell{i} = Biases;
        Betas_cell{i} = Betas;
        
        all_in_one.Mus_cell = Mus_cell;
        all_in_one.Sigmas_cell = Sigmas_cell;
        all_in_one.Biases_cell = Biases_cell;
        all_in_one.Betas_cell = Betas_cell;
        
        save(filer_1, 'all_in_one');
    end
else
    fprintf(1,'Loading all-in-one models from file... \n');
    temp = load(filer_1);
    temp = temp.all_in_one;
    Mus_cell = temp.Mus_cell;
    Biases_cell = temp.Biases_cell;
    Sigmas_cell = temp.Sigmas_cell;
    Betas_cell = temp.Betas_cell;
end

for i = 1:length(test_datas)
  
  cls_res_dir = fullfile(esvm_res_dir, test_datas{i}{1}.cls_name);

  for j = 1:length(test_datas{i})
      
      
      res = zeros(size(models));
      
      filer = sprintf('%s/%s_%s_score.mat',cls_res_dir, feat_name, test_datas{i}{j}.img_id);
      
      if ~exist(filer,'file')
          temp = cell(length(models),1);
          for m = 1:length(models)
            res_per_class = zeros(size(models{m}));
            input = test_datas{i}{j}.feature;
            standarized_inputs = (repmat(input, size(Mus_cell{m},1), 1) - Mus_cell{m} ) ./ Sigmas_cell{m};
            %replace any entry which is finite with 0
            [row, col] = find(~isfinite(standarized_inputs));
            standarized_inputs(row,col) = 0;
            res_per_class = sum(standarized_inputs  .* Betas_cell{m}, 2) + Biases_cell{m};

             [res(m), Index_J_temp(m)] = max(res_per_class);
             %[sorted,~] = sort(res_per_class);
             %res(m) = mean(sorted(1:floor(length(res_per_class)/8)));
             %res(m) = mean(sorted(1:10));
             temp{m} = res_per_class;
          end
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
          save(filer,'result');
      else 
          file = load(filer);
          result = file.result;
      end    
      
      test_datas{i}{j}.score = result.res;
      test_datas{i}{j}.index_i = result.i;
      test_datas{i}{j}.index_j = result.j;
      counter = counter + 1;

      test_data = test_datas{i}{j};
      if mod(counter,30) == 0
         fprintf(1,'Predict test images %d/%d, image_id = %s, class = %s \n', counter, ...
                                    num_test_images,test_data.img_id, test_data.cls_name);
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