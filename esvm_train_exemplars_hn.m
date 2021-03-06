function new_models = ...
    esvm_train_exemplars_hn(models, neg_set, cal_set, feat_name, params)
% Train exemplar-SVM models with hard negatives mining.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

models_root_dir = fullfile('.', params.datasets_params.results_folder,'models');
models_hn_dir = fullfile(models_root_dir, 'hard_negative');
models_dir = fullfile(models_hn_dir, feat_name);
algo_name = 'esvm';

if ~exist(models_root_dir, 'dir')
    mkdir(models_root_dir);
end

if ~exist(models_hn_dir, 'dir')
    mkdir(models_hn_dir);
end

if ~exist(models_dir, 'dir')
    mkdir(models_dir);
end
% The single file which contains all exempalr-SVM models
filer_1 = sprintf('%s/%s_models_in_matrix.mat', models_root_dir, feat_name);

new_models = cell(size(models));
num_models = 0;

for i = 1:length(models)
    num_models = num_models + numel(models{i});
end

counter = 1;

for qq = 1:length(models)
    cls_models = models{qq};
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
      
      if ~exist(filer2final,'file') && ~exist(filer_1,'file')
          fprintf(1,'Strat to train model %d/%d, model_id = %s, class = %s \n',...
                                counter, num_models,m.img_id, m.cls_name);
          % Add training set and training set's mining queue 
          m.neg_set = neg_set{qq}{i};
          m.mining_queue = esvm_initialize_mining_queue(m.neg_set);
          m.mining_params = params.training_params;
          if ~isfield(m,'cal_set')
            m.cal_set = cal_set{qq}{i}.neg_filer;
          end

          m.models_name = sprintf('%s_%s_%s_%s.mat',feat_name,algo_name,m.cls_name,m.img_id);
          m.iteration = 1;
          % Set a flag to terminate iteration when no negative support
          % vectors can be found
          m.no_negatives_found = false;

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
            m.svm_model = compact(m.svm_model);
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
          end 
      else
          if mod(counter,50) == 0
          fprintf(1,'Load model %d/%d, model_id = %s, class = %s \n', counter, ...
                                        num_models,m.img_id, m.cls_name);
          end
      end
      counter = counter + 1;
      cls_new_models{i} = filer2final;
    end
    new_models{qq} = cls_new_models;
end


