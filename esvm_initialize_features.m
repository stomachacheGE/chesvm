function [train_datas, test_datas]= esvm_initialize_features(datasets_info, feat_name, algo_name ,params)


datasets_params = params.datasets_params;
feat_params = params.features_params;

if strcmp(feat_name,'cnn')% MatConvNet
    cnn_params = feat_params.cnn_params;
    mcnpath = fullfile('.', 'lib','matconvnet-1.0-beta18','matlab');
    run(fullfile(mcnpath,'vl_setupnn.m'));
    cnn_path = fullfile('.',cnn_params.model_folder,cnn_params.model_name);
    convnet = load(cnn_path);   
end

train_datas = cell(1, length(datasets_info));
test_datas = cell(1, length(datasets_info));

for j=1:length(datasets_info)
    
    cls = datasets_info{j}.cls_name;
    feat_res_dir = fullfile('.', datasets_params.results_folder,'features');
    res_dir = fullfile('.', datasets_params.results_folder,'features', cls);


    if ~exist(feat_res_dir, 'dir')
        mkdir(feat_res_dir);
        mkdir(res_dir);
    else
        if ~exist(res_dir,'dir')
           mkdir(res_dir);
        end
    end

    img_files = [datasets_info{j}.train_image_files datasets_info{j}.test_image_files];
    img_ids = [datasets_info{j}.train_image_ids datasets_info{j}.test_image_ids];

    counter = 0;
    
    train_datas{j} = cell(1, length(datasets_info{j}.train_image_ids));
    test_datas{j} = cell(1, length(datasets_info{j}.test_image_ids));

    for i=1:length(img_ids)
        
        if strcmp(feat_name, 'hog')
            %filer = fullfile(res_dir, sprintf('%s-%s-not-resized.mat',feat_name,img_ids{i}));
            filer = fullfile(res_dir, sprintf('%s-%s-80-56.mat',feat_name,img_ids{i}));
        else
            filer = fullfile(res_dir, sprintf('%s-%s.mat',feat_name,img_ids{i}));
        end
        
        %filers = fullfile(res_dir, sprintf('cnn-%s-not-resized.mat',img_ids{i}));
        %if exist(filers,'file')
        %   delete(filers);
        %end
        
        %if exist(filer,'file')
        %   delete(filer);
        %end

        if ~exist(filer,'file')
            %fprintf(1,' Calculating features...');
            img = imread(img_files{i});
            data.img_size = [size(img,1) size(img,2)];
            if strcmp(feat_name,'cnn')
                data.feature = double(esvm_extract_cnn_feature(img, convnet, cnn_params.layer));
            else
                hog_params = feat_params.hog_params;
                img = imresize(double(img),[hog_params.height hog_params.width]);
                [data.feature data.hog_size] = esvm_extract_hog_feature(double(img), feat_params.hog_params);                     

            end

            data.cls_name = cls;
            data.label = j;
            data.img_id = img_ids{i}; 
            data.feat_filer = filer;
            counter = counter + 1;
            save(filer, 'data');
            
            if mod(counter,30) == 0
            fprintf(1, 'Extracting %s features for %s: %d/%d \n',...
                        feat_name, cls, counter, length(img_ids));
            end
        else
            %fprintf(1,' Loading features...');
            data = load(filer, 'data');
            data = data.data;
            counter = counter + 1;
            
            if mod(counter,30) == 0
            fprintf(1, 'Loading %s features for %s: %d/%d \n',...
                        feat_name, cls, counter, length(img_ids));
            end
        end
        
        
        if i <= length(datasets_info{j}.train_image_ids)
            data.img_filer = datasets_info{j}.train_image_files{i};
            train_datas{j}{i} = data;
        else
            data.img_filer = datasets_info{j}.test_image_files{i-length(datasets_info{j}.train_image_ids)};
            test_datas{j}{i-length(datasets_info{j}.train_image_ids)} = data;
        end



        if counter == length(img_ids)
            fprintf(1, ' *********Extracting %s features for %s finished. Total: %d *********\n',...
                        feat_name, cls, counter);
        end



    end
end

%concantenate all cell array contents to a cell array
%train_datas = [horzcat(train_datas{:})];
%test_datas = [horzcat(test_datas{:})];

end