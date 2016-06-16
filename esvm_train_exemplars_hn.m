function new_models = ...
    esvm_train_exemplars_hn(models, neg_set, feat_name, params)
% Train models with hard negatives mined from neg_set
% [models]: a cell array of initialized exemplar models
% [neg_set]: a virtual set of images to mine from
% [feat_name]: the name of feature used
% [params]: parameters used for training

%Return:
%[new_models]: a cell array of the filename of generated models

models_dir = fullfile('.', params.datasets_params.results_folder,'models');
algo_name = 'esvm';

if ~exist(models_dir, 'dir')
    mkdir(models_dir);
end

new_models = cell(size(models));

%get the number of models
num_models = 0;
for i = 1:length(models)
    num_models = num_models + numel(models{i});
end

counter = 1;

for qq = 1:length(models)
    
    %get the models of a specific class
    cls_models = models{qq};
    %clear cls_new_models;
    %initialize a cell array of models of a specific calss
    cls_new_models = cell(size(cls_models));
    
    for i = 1:length(cls_models)
      clear m;
      m = cls_models{i};
      
      %check whether a directory exists for storing models of a specific
      %class. If not, make it
      cls_models_dir = fullfile(models_dir, m.cls_name);
      if ~exist(cls_models_dir, 'dir')
            mkdir(cls_models_dir);
      end

      %get the filename of temporary models and final model
      filer2fill = sprintf('%s/%%s_%s_%s_%s_%s.mat',cls_models_dir,feat_name,algo_name,m.cls_name,m.img_id);
      filer2final = sprintf('%s/%s_%s_%s_%s.mat',cls_models_dir,feat_name,algo_name,m.cls_name,m.img_id);
      
      if ~exist(filer2final,'file')
          fprintf(1,'Strat to train model %d/%d, model_id = %s, class = %s \n',...
                                counter, num_models,m.img_id, m.cls_name);
          % Add training set and training set's mining queue 
          m.neg_set = neg_set{qq};
          m.mining_queue = esvm_initialize_mining_queue(m.neg_set);

          % Add mining_params, and params.dataset_params to this exemplar
          m.mining_params = params.training_params;


          % Append '-svm' to the mode to create the models name
          m.models_name = sprintf('%s_%s_%s_%s.mat',feat_name,algo_name,m.cls_name,m.img_id);
          m.iteration = 1;
          % Set a flag to terminate iteration when no negative support
          % vectors can be found
          m.no_negatives_found = false;
          
          % The mining queue is the ordering in which we process new images  
          keep_going = 1;

          while keep_going == 1

            %Get the name of the next chunk file to write
            filer2 = sprintf(filer2fill,num2str(m.iteration));

            if ~isfield(m,'mining_stats')
              total_mines = 0;
            else
              total_mines = sum(cellfun(@(x)x.total_mines,m.mining_stats));
            end
            m.total_mines = total_mines;
            %pass the model and actually train it
            m = esvm_mine_train_iteration(m, feat_name, m.mining_params.training_function);

            if ((total_mines >= params.training_params.train_max_mined_images) || ...
                  (isempty(m.mining_queue)) || ...
                  (m.iteration == params.training_params.train_max_mine_iterations)||...
                  (m.no_negatives_found))
              
              keep_going = 0;      
              %bump up filename to final file
              filer2 = filer2final;
            end

            %HACK: remove neg_set which causes save issue when it is a
            %cell array of function pointers
            msave = m;
            m = rmfield(m,'neg_set');
            save(filer2,'m');
            m = msave;
            
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
      else
          if mod(counter,50) == 0
          fprintf(1,'Load model %d/%d, model_id = %s, class = %s \n', counter, ...
                                        num_models,m.img_id, m.cls_name);
          end
          
          %clear m;
          %load m from model file
          %load(filer2final);
      end
      
      counter = counter + 1;
      %append the generated model filename to cls_new_models
      cls_new_models{i} = filer2final;
    end
    new_models{qq} = cls_new_models;
end


