function datasets = esvm_get_datasets_info(params)

data_dir = fullfile('.', params.img_folder, params.dataset_dir);

if exist([data_dir '/datasets_info.mat'], 'file')
    load([data_dir '/datasets_info.mat']);
    for i = 1:length(datasets)
    fprintf(1,'Find image class %s, #train:%d; #val:%d, #test:%d \n', datasets{i}.cls_name,...
                                          length(datasets{i}.train_image_ids),  length(datasets{i}.val_image_ids), length(datasets{i}.test_image_ids));
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
        
        %randomize orderings
        myRandomize;
        ordering = randperm(length(train_image_names));
        %take 1/10 as validation
        num_val = length(ordering) / 10;
        datasets{i}.val_image_files = datasets{i}.train_image_files(ordering(1:num_val));
        datasets{i}.train_image_files = datasets{i}.train_image_files(ordering(num_val:end));
        datasets{i}.val_image_ids = datasets{i}.train_image_ids(ordering(1:num_val));
        datasets{i}.train_image_ids = datasets{i}.train_image_ids(ordering(num_val:end));

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
                                    
        %randomize orderings
        myRandomize;
        ordering = randperm(length(test_image_names));
        num_val = length(datasets{i}.val_image_files);
        datasets{i}.val_image_files = horzcat(datasets{i}.val_image_files,datasets{i}.test_image_files(ordering(1:num_val)));
        datasets{i}.train_image_files = datasets{i}.test_image_files(ordering(num_val:end));
        datasets{i}.val_image_ids = horzcat(datasets{i}.val_image_ids,datasets{i}.test_image_ids(ordering(1:num_val)));
        datasets{i}.train_image_ids = datasets{i}.test_image_ids(ordering(num_val:end));

        fprintf(1,'Find image class %s, #train:%d; #val:%d, #test:%d \n', datasets{i}.cls_name,...
                                          length(datasets{i}.train_image_ids),  length(datasets{i}.val_image_ids), length(datasets{i}.test_image_ids));


    end
    
 %   for i=1:length(datasets)
 %       q = 1:length(datasets)
        
 %       for j = 1:length(datasets{i})
            
    

    %store the gt .txt file to disk
    filer = sprintf([data_dir '/datasets_info.mat']);
    fprintf(1,'Saving datasets info into file %s\n',filer);
    save(filer, 'datasets');
end
end