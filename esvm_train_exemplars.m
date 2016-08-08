function new_models = esvm_train_exemplars(models, train_set, cal_set, feat_name, params)
% Train exemplar-SVM models without hard negatives mining.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

models_root_dir = fullfile('.', params.datasets_params.results_folder,'models');
models_hn_dir = fullfile(models_root_dir, 'wo_hard_negative');
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
filer_1 = sprintf('%s/%s_models_in_matrix_wo_hn.mat', models_root_dir, feat_name);

new_models = cell(size(models));
num_models = 0;

for i = 1:length(models)
    num_models = num_models + numel(models{i});
end

counter = 1;

for qq = 1:length(models)
    cls_models = models{qq};
    clear cls_new_models;
    cls_new_models = cell(size(cls_models));
    
    for i = 1:length(cls_models)
      clear m;
      m = cls_models{i};
      
      cls_models_dir = fullfile(models_dir, m.cls_name);
      if ~exist(cls_models_dir, 'dir')
            mkdir(cls_models_dir);
      end
      % exempalr-SVM model file name
      filer2final = sprintf('%s/%s_%s_%s_%s.mat',cls_models_dir,feat_name,algo_name,m.cls_name,m.img_id);
      
      if ~exist(filer2final,'file') && ~exist(filer_1,'file')
          fprintf(1,'Strat to train model using linear svm %d/%d, model_id = %s, class = %s \n', counter, ...
                                        num_models,m.img_id, m.cls_name);
          m.train_set = train_set{qq}{i};
          if ~isfield(m,'cal_set')
              m.cal_set = cal_set{qq}{i}.neg_filer;
          end
          m.models_name = sprintf('%s_%s_%s_%s.mat',feat_name,algo_name,m.cls_name,m.img_id);

          neg_feature_filers = train_set{qq}{i}.feat_filers;
          neg_img_filers = train_set{qq}{i}.img_filers;
          neg_features = cell(1,length(neg_feature_filers));

          for filer_i = 1:length(neg_feature_filers)
               temp = load(neg_feature_filers{filer_i});
               neg_features{filer_i} = temp.data.feature;
          end
          neg_features = [vertcat(neg_features{:})];                   
          train_features = vertcat(m.x, neg_features);

          neg_labels = ones(length(neg_feature_filers),1);
          neg_labels = -neg_labels;
          train_labels = vertcat(1, neg_labels) ;
          % Train exemplar-SVM with features and labels.
          m = linSVM_train_exemplar(m, train_features, train_labels, params);

          %HACK: remove train_set which causes save issue when it is a
          %cell array of function pointers
          msave = m;
          m = rmfield(m,'train_set');
          m.svm_model = compact(m.svm_model);
          save(filer2final,'m');
          m = msave;
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
end


    


