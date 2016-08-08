function datasets = esvm_get_datasets_info(params)
% Get the image set information and store it to datasets_info.mat.
% Note that this file should be manually deleted if any modifications 
% are made to the image set.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

data_dir = fullfile('.', params.img_folder, params.dataset_dir);

% If the dasesets_info.mat file exists, load it.
% Otherwise, retrieve the information and store it.
if exist([data_dir '/datasets_info.mat'], 'file')
    load([data_dir '/datasets_info.mat']);
    for i = 1:length(datasets)
    fprintf(1,'Find image class %s, #train:%d; #test:%d \n', datasets{i}.cls_name,...
                                          length(datasets{i}.train_image_ids), length(datasets{i}.test_image_ids));
    end
else
    
    d = dir(data_dir);
    d = d(3:end); %remove . and .. dirs
    d = d(cat(1, d.isdir)); % remove non dirs

    datasets = cell(1,length(d)); 
    counter = 0;

    %loop over each class folder
    for i=1:length(d) 
        cls_name = d(i).name;
        datasets{i}.cls_name = cls_name;

        %start with train images
        train_img_dir = dir(fullfile('.', params.img_folder, ...
                                       params.dataset_dir, cls_name,...
                                       'train', ['*.' params.file_ext]));
        if params.rename %rename image name
          for id = 1:length(train_img_dir)
            old_file = fullfile('.', params.img_folder, ...
                                       params.dataset_dir, cls_name,...
                                       'train', train_img_dir(id).name);
            new_file = fullfile('.', params.img_folder, ...
                                       params.dataset_dir, cls_name,...
                                       'train', sprintf('%06d.%s', counter, params.file_ext));    
            if ~strcmp(old_file,new_file)
                movefile(old_file, new_file);  
            end
            counter = counter + 1;
          end

          train_img_dir = dir(fullfile('.', params.img_folder, ...
                                       params.dataset_dir, cls_name,...
                                       'train', ['*.' params.file_ext]));     
        end

        train_image_names = {train_img_dir(:).name};
        datasets{i}.train_image_files = cellfun(@(x) fullfile('.', params.img_folder, ...
                                       params.dataset_dir, cls_name,'train', x), train_image_names,...
                                       'UniformOutput', false);
        datasets{i}.train_image_ids = cellfun(@(x) x(1:end-length(params.file_ext)-1), train_image_names,...
                                        'UniformOutput', false); 
        %then deal with test images
        test_img_dir = dir(fullfile('.', params.img_folder, ...
                                       params.dataset_dir, cls_name,...
                                       'test', ['*.' params.file_ext]));
        if params.rename %rename image name
          for id = 1:length(test_img_dir)
            old_file = fullfile('.', params.img_folder, ...
                                       params.dataset_dir, cls_name,...
                                       'test', test_img_dir(id).name);
            new_file = fullfile('.', params.img_folder, ...
                                       params.dataset_dir, cls_name,...
                                       'test', sprintf('%06d.%s', counter, params.file_ext));                   
            if ~strcmp(old_file,new_file)
                movefile(old_file, new_file);      
            end
            counter = counter + 1;          
          end
          test_img_dir = dir(fullfile('.', params.img_folder, ...
                                       params.dataset_dir, cls_name,...
                                       'test', ['*.' params.file_ext]));     
        end

        test_image_names = {test_img_dir(:).name};
        datasets{i}.test_image_files = cellfun(@(x) fullfile('.', params.img_folder, ...
                                       params.dataset_dir, cls_name,'test', x), test_image_names,...
                                       'UniformOutput', false);
        datasets{i}.test_image_ids = cellfun(@(x) x(1:end-length(params.file_ext)-1), test_image_names,...
                                        'UniformOutput', false); 

        fprintf(1,'Find image class %s, #train:%d; #test:%d \n', datasets{i}.cls_name,...
                                          length(datasets{i}.train_image_ids), length(datasets{i}.test_image_ids));

    end   
    %store the datasets_info.mat file to disk
    filer = sprintf([data_dir '/datasets_info.mat']);
    fprintf(1,'Saving datasets info into file %s\n',filer);
    save(filer, 'datasets');
end
end