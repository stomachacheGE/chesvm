function esvm_show_classification(cls_idx, idx, feat_name, hard_negative, datasets_info, params)
%% THIS FUNCTION IS DEPRECIATED AND PROBABLY NOT WORK.
% Show the retrieved most-similar-exempalrs for a test image which is
% specified by cls_idx and idx.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/
classes = cellfun(@(x) x.cls_name, datasets_info, 'UniformOutput', false);

num_imgs = cellfun(@(x) length(x.train_image_ids) + length(x.test_image_ids), datasets_info, 'UniformOutput', false);
num_imgs = cell2mat(num_imgs);

offset(1) = 0;
for i=1:length(num_imgs)
    offset(i+1) = sum(num_imgs(1:i));
end

dataset_folder = sprintf('./%s/%s', params.datasets_params.img_folder, ...
                           params.datasets_params.dataset_dir);
img_foler = sprintf('./%s/%s/%s/test', params.datasets_params.img_folder, ...
                           params.datasets_params.dataset_dir, classes{cls_idx});                      
esvm_res_dir = sprintf('./%s/classifications/esvm',params.datasets_params.results_folder);

id = offset(cls_idx) + length(datasets_info{cls_idx}.train_image_ids) + idx - 1;

figure(1);
e_filer = sprintf('%s/%06d.%s',img_foler,id, params.datasets_params.file_ext);
imshow(e_filer);

figure(20);
width = params.features_params.hog_params.width;
height = params.features_params.hog_params.height;
cls_res_dir = fullfile(esvm_res_dir, classes{cls_idx});
if hard_negative
    res_filer = sprintf('%s/%s_%06d_score.mat',cls_res_dir, feat_name, id);
else
    res_filer = sprintf('%s/%s_%06d_score_wo_hn.mat',cls_res_dir, feat_name, id);
end

result = load(res_filer);
result = result.result;
Index_I = result.i;

score = result.scores{Index_I};                       
[sorted_scores, indexes] = sort(score, 'descend');

for mm=1:3
    for n=1:4
        which = (mm-1)*4+n;
        filer = sprintf('%s/%s/%s/%06d.%s',dataset_folder, classes{Index_I}, 'train', offset(Index_I)+indexes(which)-1, params.datasets_params.file_ext);
        img_temp = imread(filer);
        img_temp = imresize(img_temp,[height width]);
        subplot(3,4,which); subimage(img_temp);
        axis off;
        title(sprintf('s=%f',sorted_scores(which)));
    end
end
num_postives = sum(score>0);
subtitle(sprintf('num positives=%d, predicted_label=%d', num_postives, Index_I));
end
