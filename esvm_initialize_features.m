function [train_datas, test_datas]= esvm_initialize_features(datasets_info, feat_name, algo_name ,params)
% Extract features out of images in the trainig set and store them to disk.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

datasets_params = params.datasets_params;
feat_params = params.features_params;

% Load pre-trained CNN model
if strcmp(feat_name,'cnn') || strcmp(feat_name,'cnnhog')% MatConvNet
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
    res_feat_dir = fullfile('.', datasets_params.results_folder,'features', feat_name);
    res_dir = fullfile('.', datasets_params.results_folder,'features', feat_name, cls);


    if ~exist(feat_res_dir, 'dir')
        mkdir(feat_res_dir);
    end
    
    if ~exist(res_feat_dir, 'dir')
    mkdir(res_feat_dir);
    end

    if ~exist(res_dir,'dir')
       mkdir(res_dir);
    end


    img_files = [datasets_info{j}.train_image_files datasets_info{j}.test_image_files];
    img_ids = [datasets_info{j}.train_image_ids datasets_info{j}.test_image_ids];

    counter = 0;
    
    train_datas{j} = cell(1, length(datasets_info{j}.train_image_ids));
    test_datas{j} = cell(1, length(datasets_info{j}.test_image_ids));

    for i=1:length(img_ids)
        filer = fullfile(res_dir, sprintf('%s-%s.mat',feat_name,img_ids{i}));
        % If feature file does not exist, extract and store it.
        % Otherwise, load it to the program.
        if ~exist(filer,'file')
            %fprintf(1,' Calculating features...');
            img = imread(img_files{i});
            data.img_size = [size(img,1) size(img,2)];
            if strcmp(feat_name,'cnn')
                data.feature = double(esvm_extract_cnn_feature(img, convnet, cnn_params.layer));
            elseif strcmp(feat_name,'hog')
                hog_params = feat_params.hog_params;
                img = imresize(double(img),[hog_params.height hog_params.width]);
                [data.feature data.hog_size] = esvm_extract_hog_feature(double(img), feat_params.hog_params);                     
            elseif strcmp(feat_name,'cnnhog')
                cnn_feature = double(esvm_extract_cnn_feature(img, convnet, cnn_params.layer));
                hog_params = feat_params.hog_params;
                img = imresize(double(img),[hog_params.height hog_params.width]);
                [hog_feature data.hog_size] = esvm_extract_hog_feature(double(img), feat_params.hog_params);   
                data.feature = horzcat(cnn_feature, hog_feature);
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
        
        % add corresponding image filename to the feature struct.
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

end