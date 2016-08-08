function [hog_model_hn, hog_model_wo_hn, hog_cal_matrix_hn, hog_cal_matrix_wo_hn,...
          cnn_model_hn, cnn_model_wo_hn, cnn_cal_matrix_hn, cnn_cal_matrix_wo_hn,...
          cnnhog_model_hn, cnnhog_model_wo_hn, cnnhog_cal_matrix_hn, cnnhog_cal_matrix_wo_hn] = esvm_gui_initialize
% Load trianed models.
% This function is used for the graphical user interface.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

params = esvm_get_default_params;
classifi_res_dir = fullfile('.', params.datasets_params.results_folder,'classifications');
models_dir = fullfile('.', params.datasets_params.results_folder,'models');
esvm_res_root_dir = fullfile(classifi_res_dir, 'esvm');

esvm_res_hn_cal_dir = fullfile(esvm_res_root_dir, 'hard_negative', 'calibration');
esvm_res_wo_hn_cal_dir = fullfile(esvm_res_root_dir, 'wo_hard_negative', 'calibration');

esvm_wo_hn_cal_mat_hog = fullfile(esvm_res_wo_hn_cal_dir, 'hog', 'hog_esvm_calibration_matrix.mat');
esvm_wo_hn_cal_mat_cnn = fullfile(esvm_res_wo_hn_cal_dir, 'cnn', 'cnn_esvm_calibration_matrix.mat');
esvm_wo_hn_cal_mat_cnnhog = fullfile(esvm_res_wo_hn_cal_dir, 'cnnhog', 'cnnhog_esvm_calibration_matrix.mat');
esvm_hn_cal_mat_hog = fullfile(esvm_res_hn_cal_dir, 'hog', 'hog_esvm_calibration_matrix.mat');
esvm_hn_cal_mat_cnn = fullfile(esvm_res_hn_cal_dir, 'cnn', 'cnn_esvm_calibration_matrix.mat');
esvm_hn_cal_mat_cnnhog = fullfile(esvm_res_hn_cal_dir, 'cnnhog', 'cnnhog_esvm_calibration_matrix.mat');

hog_cal_matrix_wo_hn = load_calibration_mat(esvm_wo_hn_cal_mat_hog);
cnn_cal_matrix_wo_hn = load_calibration_mat(esvm_wo_hn_cal_mat_cnn);
cnnhog_cal_matrix_wo_hn = load_calibration_mat(esvm_wo_hn_cal_mat_cnnhog);
hog_cal_matrix_hn = load_calibration_mat(esvm_hn_cal_mat_hog);
cnn_cal_matrix_hn = load_calibration_mat(esvm_hn_cal_mat_cnn);
cnnhog_cal_matrix_hn = load_calibration_mat(esvm_hn_cal_mat_cnnhog);

hog_model_hn = load_model(fullfile(models_dir,'hog_models_in_matrix.mat'));
hog_model_wo_hn = load_model(fullfile(models_dir,'hog_models_in_matrix_wo_hn.mat'));
cnn_model_hn = load_model(fullfile(models_dir,'cnn_models_in_matrix.mat'));
cnn_model_wo_hn = load_model(fullfile(models_dir,'cnn_models_in_matrix_wo_hn.mat'));
cnnhog_model_hn = load_model(fullfile(models_dir,'cnnhog_models_in_matrix.mat'));
cnnhog_model_wo_hn = load_model(fullfile(models_dir,'cnnhog_models_in_matrix_wo_hn.mat'));

end

function result = load_calibration_mat(path)
    if exist(path, 'file')
      file = load(path);
      result = file.cal_mat;
    else
       result = NaN;
    end
end

function model = load_model(path)
    if exist(path, 'file')
      temp = load(path);
      model = temp.all_in_one;
    else
       model = NaN;
    end
end