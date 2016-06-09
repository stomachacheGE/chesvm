%function prediction = esvm_predict(models, test_datas, feat_name, params)



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

for i = 1:length(test_datas)
  
  cls_res_dir = fullfile(esvm_res_dir, test_datas{i}{1}.cls_name);

  for j = 1:length(test_datas{i})
      
      model_counter = 0;
      
      res = zeros(size(models));
      
      filer = sprintf('%s/%s_%s_score.mat',cls_res_dir, feat_name, test_datas{i}{j}.img_id);
      
      if ~exist(filer,'file')
          for m = 1:length(models)

             res_per_class = zeros(size(models{m}));

             for n = 1:length(models{m}) 
                  %model = models{m}{n};
                  model = load(models{m}{n});
                  model = model.m;
                  if strcmp(feat_name,'cnn');
                      x_test = test_datas{i}{j}.feature;
                      %fprintf(1,'loading cnn feature\n');
                      %x = x.data.feature;
                  else
                      img = imread(test_datas{i}{j}.img_filer);
                      img = imresize(double(img), model.img_size);
                      %fprintf(1,'%d size img is [%d %d] \n',index, size(img,1),size(img,2));
                      [x_test, ~] = params.features_params.hog_extractor(img);
                      %fprintf(1,'loading hog feature\n');
                  end  
                  res_per_class(n) = model.w * x_test' - model.b;

                  model_counter = model_counter + 1;

                    if mod(model_counter,30) == 0
                    fprintf(1,'Predicting test image %s on models: %d/%d \n',  ...
                                                    test_datas{i}{j}.img_id,model_counter, num_models);
                    end
             end   

             res(m) = max(res_per_class);

          end
          %normalize sum of scores to 1
          %res = res/sum(res);
          save(filer,'res');
      else 
          file = load(filer);
          res = file.res;
      end    
      
      test_datas{i}{j}.score = res;
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
sum_scores_per_test_data = sum(scores,2);
sum_scores_per_test_data = repmat(sum_scores_per_test_data,1,size(scores,2));

scores = scores./sum_scores_per_test_data;
[~, indexes] = max(scores,[],2);

prediction.ids = indexes;
prediction.prob = scores;
