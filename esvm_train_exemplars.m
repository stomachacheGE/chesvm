function new_models = esvm_train_exemplars(models, train_datas, train_set, algo_name, feat_name, params)
% Train models with hard negatives mined from train_set
% [models]: a cell array of initialized exemplar models
% [train_set]: a virtual set of images to mine from
% [params]: localization and training parameters


models_dir = fullfile('.', params.datasets_params.results_folder,'models');


if ~exist(models_dir, 'dir')
    mkdir(models_dir);
end


new_models = cell(size(models));
%{
randomize chunk orderings
if CACHE_FILE == 1
  myRandomize;
  ordering = randperm(length(models));
else
  ordering = 1:length(models);
end

models = models(ordering);
%}
num_models = 0;
for i = 1:length(models)
    num_models = num_models + numel(models{i});
end

counter = 1;

for qq = 1:length(models)
    

    cls_models = models{qq};
    %newmodels{i} = cell(length(models), 1);
    clear cls_new_models;
    cls_new_models = cell(size(cls_models));
    
    for i = 1:length(cls_models)
      %filer = '';
      clear m;
      m = cls_models{i};

      %{
      [complete_file] = sprintf('%s/%s.mat',models_dir,m.name);
      [basedir, basename, ext] = fileparts(complete_file);
      filer2fill = sprintf('%s/%%s.%s.mat',basedir,basename);
      filer2final = sprintf('%s/%s.mat',basedir,basename);  
      %}
      cls_models_dir = fullfile(models_dir, m.cls_name);
      if ~exist(cls_models_dir, 'dir')
            mkdir(cls_models_dir);
      end

      
      %filer2fill = sprintf('%s/%%s_%s_%s_%s_%s_wo_hn.mat',cls_models_dir,feat_name,algo_name,m.cls_name,m.img_id);
      filer2final = sprintf('%s/%s_%s_%s_%s_wo_hn.mat',cls_models_dir,feat_name,algo_name,m.cls_name,m.img_id);
      
      if ~exist(filer2final,'file')
          fprintf(1,'Strat to train model using linear svm %d/%d, model_id = %s, class = %s \n', counter, ...
                                        num_models,m.img_id, m.cls_name);
          % Add training set and training set's mining queue 
          %fprintf(1,'q is %d\n',qq);
          %fprintf(1,'train_set size is [%d %d]\n', size(train_set,1),size(train_set,2));
          m.train_set = train_set{qq};
          %m.mining_queue = esvm_initialize_mining_queue(m.train_set);

          % Add mining_params, and params.dataset_params to this exemplar
          %m.mining_params = params.training_params;


          % Append '-svm' to the mode to create the models name
          m.models_name = sprintf('%s_%s_%s_%s.mat',feat_name,algo_name,m.cls_name,m.img_id);
          %m.iteration = 1;

          % The mining queue is the ordering in which we process new images  
          %keep_going = 1;
          %{
          while keep_going == 1

            %Get the name of the next chunk file to write
            filer2 = sprintf(filer2fill,num2str(m.iteration));

            if ~isfield(m,'mining_stats')
              total_mines = 0;
            else
              total_mines = sum(cellfun(@(x)x.total_mines,m.mining_stats));
            end
            m.total_mines = total_mines;
            m = esvm_mine_train_iteration(m, feat_name, m.mining_params.training_function);

            if ((total_mines >= params.training_params.train_max_mined_images) || ...
                  (isempty(m.mining_queue))) || ...
                  (m.iteration == params.training_params.train_max_mine_iterations)

              keep_going = 0;      
              %bump up filename to final file
              filer2 = filer2final;
            end
          %}
          
            neg_feature_filers = train_set{qq}{i}.feat_filers;
            neg_img_filers = train_set{qq}{i}.img_filers;
            neg_features = cell(1,length(neg_feature_filers));
            

            for filer_i = 1:length(neg_feature_filers)
               if strcmp(feat_name,'cnn')
                 temp = load(neg_feature_filers{filer_i});
                 neg_features{filer_i} = temp.data.feature;
               else
                 %{
                 img = imread(neg_img_filers{filer_i});
                 img = imresize(double(img), m.img_size);
                 %fprintf(1,'%d size img is [%d %d] \n',index, size(img,1),size(img,2));
                 [temp, ~] = params.training_params.hog_extractor(img);
                 neg_features{filer_i} = temp;
                 %}
                 temp = load(neg_feature_filers{filer_i});
                 neg_features{filer_i} = temp.data.feature;
               end
            end
            neg_features = [vertcat(neg_features{:})];                   
            train_features = vertcat(m.x, neg_features);
            
            neg_labels = ones(length(neg_feature_filers),1);
            neg_labels = -neg_labels;
            train_labels = vertcat(1, neg_labels) ;
            
            m = linSVM_train_exemplar(m, train_features, train_labels, params);
            
            datas = cellfun(@(x)x.feature, train_datas{qq},'UniformOutput',false);
            datas_1 = [vertcat(datas{:})];
            %normalize train_features
            %d_mean = mean(datas_1, 1);
            %d_std = std(datas_1, 1);
            %normalized_datas_1 = (datas_1 - repmat(d_mean, size(datas_1,1),1)) ./ repmat(d_std, size(datas_1,1),1);
            %fprintf(1,'dimension of wex is [%d %d]\n', size(wex,1),size(wex,2));
            %fprintf(1,'dimension of x is [%d %d]\n', size(m.x,1),size(m.x,2));
            %scores = wex*normalized_datas_1' - b;
            [predicted_labels, scores_both] = predict(m.svm_model, datas_1);
            predicted_scores = scores_both(:,2);
            [sorted_scores, ~] = sort(predicted_scores, 'descend');
            m.self_bias = sorted_scores(1) - sorted_scores(2);

      
            
            %HACK: remove train_set which causes save issue when it is a
            %cell array of function pointers
            msave = m;
            m = rmfield(m,'train_set');
            save(filer2final,'m');
            m = msave;

            %{
            %delete old files
            if m.iteration > 1
              for q = 1:m.iteration-1
                filer2old = sprintf(filer2fill,num2str(q));
                if fileexists(filer2old)
                    delete(filer2old);
                end
              end
            end

            if keep_going==0
              fprintf(1,' ### End of training for this model... \n');
              break;
            end

            m.iteration = m.iteration + 1;
          end %iteratiion
          %fprintf(1,'Training model %d/%d, model_id = %d, class = %s finished \n', counter, ...
                                 %num_models,m.img_id, m.cls_name);
        %}
      else
          if mod(counter,50) == 0
          fprintf(1,'Load model %d/%d, model_id = %s, class = %s \n', counter, ...
                                        num_models,m.img_id, m.cls_name);
          end
          %model = load(filer2final);
          %clear m;
          %m = model.m;
      end
      counter = counter + 1;
      %cls_new_models{i} = m;
      cls_new_models{i} = filer2final;
    end
    new_models{qq} = cls_new_models;
end
end


    


