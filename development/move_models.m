function move_models(models, algo_name, feat_name, params)
% Train models with hard negatives mined from train_set
% [models]: a cell array of initialized exemplar models
% [train_set]: a virtual set of images to mine from
% [params]: localization and training parameters

old_folder = 'results_VOC_3_validation';
folder = 'results_VOC_4';
source_models_dir = fullfile('.', old_folder,'models');
des_models_dir = fullfile('.',folder,'models');

if ~exist(des_models_dir, 'dir')
    mkdir(des_models_dir);
end


num_models = 0;
for i = 1:length(models)
    num_models = num_models + numel(models{i});
end


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
      source_cls_models_dir = fullfile(source_models_dir, m.cls_name);
      des_cls_models_dir = fullfile(des_models_dir, m.cls_name);
      if ~exist(des_cls_models_dir, 'dir')
            mkdir(des_cls_models_dir);
      end

      
      %filer2fill = sprintf('%s/%%s_%s_%s_%s_%s_wo_hn.mat',cls_models_dir,feat_name,algo_name,m.cls_name,m.img_id);
      source_filer2final = sprintf('%s/%s_%s_%s_%s_wo_hn.mat',source_cls_models_dir,feat_name,algo_name,m.cls_name,m.img_id);
      des_filer2final = sprintf('%s/%s_%s_%s_%s_wo_hn.mat',des_cls_models_dir,feat_name,algo_name,m.cls_name,m.img_id);
      movefile(source_filer2final, des_filer2final);
    end

end
end


    


