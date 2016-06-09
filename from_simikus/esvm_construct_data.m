function [train_feat_files, test_feat_files]= esvm_initialize_features(cls_info, feat_name, params)

cls = cls_info.cls_name;

datasets_params = params.datasets_params;
feat_params = params.features_params;

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

img_files = [cls_info.train_image_files cls_info.test_image_files];
img_ids = [cls_info.train_image_ids cls_info.test_image_ids];

if strcmp(feat_name,'cnn')% MatConvNet
    cnn_params = feat_params.cnn_params;
    mcnpath = fullfile('.', 'lib','matconvnet-1.0-beta18','matlab');
    run(fullfile(mcnpath,'vl_setupnn.m'));
    cnn_path = fullfile('.',cnn_params.model_folder,cnn_params.model_name);
    convnet = load(cnn_path);   
end


counter = 0;
train_feat_files = cell(1, length(cls_info.train_image_ids));
test_feat_files = cell(1, length(cls_info.test_image_ids));

for i=1:length(img_ids)
    filer = fullfile(res_dir, sprintf('%s-%s.mat',feat_name,img_ids{i}));
    
    if i <= length(cls_info.train_image_ids)
        train_feat_files{i} = filer;
    else
        test_feat_files{i-length(cls_info.train_image_ids)} = filer;
    end
    
    if ~exist(filer,'file')
        img = imread(img_files{i});
        if strcmp(feat_name,'cnn')
            feature = esvm_extract_cnn_feature(img, convnet, cnn_params.layer);
            counter = counter + 1;
            save(filer, 'feature');
        end
    else
        counter = counter + 1;
    end
    
    if mod(counter,30) == 0
        fprintf(1, 'Extracting %s features for %s: %d/%d \n',...
                    feat_name, cls, counter, length(img_ids));
    end
    
    if counter == length(img_ids)
        fprintf(1, ' *********Extracting %s features for %s finished. Total: %d *********\n',...
                    feat_name, cls, counter);
    end
    
    
    
end



end