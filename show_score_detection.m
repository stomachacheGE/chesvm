params = esvm_get_default_params;
class = {'aeroplane', 'bicycle','bird', 'bus'};
i = 4;
for id = 2455:2470
    offset = [0, 591, 1281,2226];
    new_file = sprintf('./%s/%s/%s/%s', params.datasets_params.img_folder, ...
                               params.datasets_params.dataset_dir, class{i},'test');
                               %'train', sprintf('%s.%s', params.datasets_params.file_ext));
    result = load(sprintf('/home/liangfu/Documents/exampler_svm/cnn_students_1/results_VOC_2/classifications/esvm_1/%s/hog_%06d_score.mat', class{i}, id));
    result = result.result;
    %score = result.scores{i};
    %[~, Index_I] = max(score);
    figure(3);
    e_filer = sprintf('%s/%06d.%s',new_file,id, params.datasets_params.file_ext);
    imshow(e_filer);


    %sorted_scores_1 = sorted_scores(1:12,:);
    %indexes_1 = sorted_scores(1:12,:);



              for m = 1:4

                 res_per_class = result.scores{m};



                 [res(m), Index_J_temp(m)] = max(res_per_class);
                 %[sorted,~] = sort(res_per_class);
                 %res(m) = mean(sorted(1:floor(length(res_per_class)/8)));
                 %res(m) = mean(sorted(1:10));
                 temp{m} = res_per_class;
              end

    pos_score_idx = find(res>0);
        [~, Index_I] = max(res);
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
    [~, Index_I_1] = max(res);


      Index_J = Index_J_temp(Index_I);

    figure(4);
    new_file_2 = sprintf('./%s/%s', params.datasets_params.img_folder, ...
                               params.datasets_params.dataset_dir);
    score = result.scores{Index_I};                       
    [sorted_scores, indexes] = sort(score, 'descend');
      for mm=1:3
        for n=1:4
            which = (mm-1)*4+n;
            filer = sprintf('%s/%s/%s/%06d.%s',new_file_2, class{Index_I}, 'train', offset(Index_I)+indexes(which)-1, params.datasets_params.file_ext);
            img_temp = imread(filer);
            img_temp = imresize(img_temp,[56 80]);
            subplot(3,4,which); subimage(img_temp);
            axis off;
            title(sprintf('s=%f',sorted_scores(which)));
        end
    end
    num_postives = sum(score>0);
    subtitle(sprintf('num positives=%d, predicted_label=%d', num_postives, Index_I));
end