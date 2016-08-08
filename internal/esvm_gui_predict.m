function predictions = esvm_gui_predict(test_img, feat_name, hard_negative, calibration, handles)
% Predict for a single test image using pre-trained models.
% This function is used for the graphical user interface.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

params = handles.params;
classifi_res_dir = fullfile('.', params.datasets_params.results_folder,'classifications');

% Get img_id offset    
classes = cellfun(@(x) x.cls_name, handles.datasets_info, 'UniformOutput', false);

num_imgs = cellfun(@(x) length(x.train_image_ids) + length(x.test_image_ids), handles.datasets_info, 'UniformOutput', false);
num_imgs = cell2mat(num_imgs);

offset(1) = 0;
for i=1:length(num_imgs)
    offset(i+1) = sum(num_imgs(1:i));
end

% choose the correct model
if strcmp(feat_name,'hog') && hard_negative
    temp = handles.hog_model_hn;
    cal_matrix = handles.hog_cal_matrix_hn;
elseif (strcmp(feat_name,'hog') && ~hard_negative)
    temp = handles.hog_model_wo_hn;
    cal_matrix = handles.hog_cal_matrix_wo_hn;
elseif (strcmp(feat_name,'cnn') && hard_negative)
    temp = handles.cnn_model_hn;
    cal_matrix = handles.cnn_cal_matrix_hn;
elseif (strcmp(feat_name,'cnn') && ~hard_negative)
    temp = handles.cnn_model_wo_hn;
    cal_matrix = handles.cnn_cal_matrix_wo_hn;
elseif (strcmp(feat_name,'cnnhog') && hard_negative)
    temp = handles.cnnhog_model_hn;
    cal_matrix = handles.cnnhog_cal_matrix_hn;
elseif (strcmp(feat_name,'cnnhog') && ~hard_negative)
    temp = handles.cnnhog_model_wo_hn;
    cal_matrix = handles.cnnhog_cal_matrix_wo_hn;
end

Mus_cell = temp.Mus_cell;
Biases_cell = temp.Biases_cell;
Sigmas_cell = temp.Sigmas_cell;
Betas_cell = temp.Betas_cell;

feat_params = params.features_params;
img = imread(test_img);
% extract feature out of the test image
if strcmp(feat_name,'cnn')
    cnn_params = feat_params.cnn_params;
    feature = double(esvm_extract_cnn_feature(img, handles.convnet, cnn_params.layer));
elseif strcmp(feat_name,'hog')
    hog_params = feat_params.hog_params;
    img = imresize(double(img),[hog_params.height hog_params.width]);
    [feature, ~] = esvm_extract_hog_feature(double(img), feat_params.hog_params); 
elseif strcmp(feat_name,'cnnhog')
    cnn_params = feat_params.cnn_params;
    cnn_feature = double(esvm_extract_cnn_feature(img, handles.convnet, cnn_params.layer));
    hog_params = feat_params.hog_params;
    img = imresize(double(img),[hog_params.height hog_params.width]);
    [hog_feature, ~] = esvm_extract_hog_feature(double(img), feat_params.hog_params);   
    feature = horzcat(cnn_feature, hog_feature);
            
end
                      
res = zeros(1,length(Mus_cell));
  temp = cell(length(Mus_cell),1);

% get the prediction scores 
for m = 1:length(Mus_cell)
    input = feature;
    standarized_inputs = (repmat(input, size(Mus_cell{m},1), 1) - Mus_cell{m} ) ./ Sigmas_cell{m};
    %replace any entry which is infinite with 0
    [row, col] = find(~isfinite(standarized_inputs));
    standarized_inputs(row,col) = 0;
    res_per_class = sum(standarized_inputs  .* Betas_cell{m}, 2) + Biases_cell{m};
    [res(m), Index_J_temp(m)] = max(res_per_class);
    
    if calibration
        res_per_class = 1 ./ (1 + exp(- cal_matrix{m}(:,1) .* res_per_class + cal_matrix{m}(:,2)));
        [res(m), Index_J_temp(m)] = max(res_per_class);
    end
     temp{m} = res_per_class;
end

%normalize sum of scores to 1
pos_score_idx = find(res>0);

if ~isempty(pos_score_idx)
  pos_scores = res(pos_score_idx);
  res = res/sum(pos_scores);
  neg_score_idx = find(res<0);
  res(neg_score_idx) = 0;
else
  res = -res;
  res = ones(1,size(res,1)) ./ res;
  res = res/sum(res);
end 

scores = temp;
[~, res_idxs] = sort(res, 'descend'); 
predictions = cell(1,4);

% get the first four most-similar-exemplars for the first four classes
for i=1:4
  cls_score = scores{res_idxs(i)};
  [sorted_cls_score, score_idxs] = sort(cls_score,'descend');
  pred_temp = cell(1,4);
  for j = 1:4
      pred_temp{j}.cls_idx = res_idxs(i);
      pred_temp{j}.idx = score_idxs(j);
      img_id = offset(res_idxs(i))+score_idxs(j)-1;
      img_folder = sprintf('./%s/%s/%s/train', params.datasets_params.img_folder, ...
                       params.datasets_params.dataset_dir, classes{res_idxs(i)});           
      filer = sprintf('%s/%06d.%s', img_folder, img_id, params.datasets_params.file_ext);
      pred_temp{j}.img_id = img_id;
      pred_temp{j}.img_filer = filer;
      pred_temp{j}.score = sorted_cls_score(j);
  end
  predictions{i} = pred_temp;
end
end