%function prediction = esvm_predict(models, test_datas, feat_name, params)

counter = 0;

%classifi_res_dir = fullfile('.', datasets_params.results_folder,'classifications');

%if ~exist(classifi_res_dir,'dir')
    %mkdir(clss
num_test_images = 0;
for i = 1:length(test_datas)
    num_test_images = num_test_images + numel(test_datas{i});
end

num_models = 0;
for i = 1:length(models)
    num_models = num_models + numel(models{i});
end

for i = 1:length(test_datas)
  
%  cls_models_dir = fullfile(models_dir, m.cls_name);
 % if ~exist(cls_models_dir, 'dir')
 %       mkdir(cls_models_dir);
 % end

  for j = 1:length(test_datas{i})
      
      model_counter = 0;
      
      res = zeros(size(models));
      
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
                
                if mod(model_counter,50) == 0
                fprintf(1,'Predicting test image %s on models: %d/%d \n',  ...
                                                test_datas{i}{j}.img_id,model_counter, num_models);
                end
         end   
         
         %res(m) = max(res_per_class);
         [sorted,~] = sort(res_per_class);
         res(m) = mean(sorted(1:floor(length(res_per_class)/8)));
         
      end
      
      test_datas{i}{j}.score = res;
      counter = counter + 1;
      
      test_data = test_datas{i}{j};
        

        if mod(counter,10) == 0
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

%end