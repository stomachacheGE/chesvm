          qq = 1;
%          fprintf(1,'Strat to train model using linear svm %d/%d, model_id = %s, class = %s \n', counter, ...
        %                                num_models,m.img_id, m.cls_name);
          % Add training set and training set's mining queue 
          %fprintf(1,'q is %d\n',qq);
          %fprintf(1,'neg_set size is [%d %d]\n', size(neg_set,1),size(neg_set,2));
          clear m;
          x = 11;
          m = models{1}{x};
          m.neg_set = neg_set{qq};
          %m.iteration = 1;

          % The mining queue is the ordering in which we process new images  
          %keep_going = 1;
          %{
          while keep_going == 1

            %Get the name of the next chunk file to write
            filer2 = sprintf(filer2fill,num2str(m.iteration));

            if ~isfield(m,'mining_stats')
              total_mines = 0;
            else
              total_mines = sum(cellfun(@(x)x.total_mines,m.mining_stats));
            end
            m.total_mines = total_mines;
            m = esvm_mine_train_iteration(m, feat_name, m.mining_params.datasets_params.training_function);

            if ((total_mines >= params.datasets_params.training_params.datasets_params.train_max_mined_images) || ...
                  (isempty(m.mining_queue))) || ...
                  (m.iteration == params.datasets_params.training_params.datasets_params.train_max_mine_iterations)

              keep_going = 0;      
              %bump up filename to final file
              filer2 = filer2final;
            end
          %}
            feat_name = 'hog';
            neg_feature_filers = neg_set{qq}.feat_filers;
            neg_img_filers = neg_set{qq}.img_filers;
            neg_features = cell(1,length(neg_feature_filers));
            

            for filer_i = 1:length(neg_feature_filers)
               if strcmp(feat_name,'cnn')
                 temp = load(neg_feature_filers{filer_i});
                 neg_features{filer_i} = temp.data.feature;
               else
                 %{
                 img = imread(neg_img_filers{filer_i});
                 img = imresize(double(img), m.img_size);
                 %fprintf(1,'%d size img is [%d %d] \n',index, size(img,1),size(img,2));
                 [temp, ~] = params.datasets_params.training_params.datasets_params.hog_extractor(img);
                 neg_features{filer_i} = temp;
                 %}
                 temp = load(neg_feature_filers{filer_i});
                 neg_features{filer_i} = temp.data.feature;
               end
            end
            neg_features = [vertcat(neg_features{:})];                   
            train_features = vertcat(m.x, neg_features);
            
            neg_labels = ones(length(neg_feature_filers),1);
            neg_labels = -neg_labels;
            train_labels = vertcat(1, neg_labels) ;
            
            %m = linSVM_train_exemplar(m, train_features, train_labels, params.datasets_params);
            
            %normalize train_features
            %t_mean = mean(train_features, 1);
            %t_std = std(train_features, 1);
            %normalized_train_features = (train_features - repmat(t_mean, size(train_features,1),1)) ./ repmat(t_std, size(train_features,1),1);
            
            new_file = sprintf('./%s/%s/%s/train', params.datasets_params.img_folder, ...
                                       params.datasets_params.dataset_dir, m.cls_name);
                                       %'train', sprintf('%s.%s', params.datasets_params.file_ext)); 
            figure(1)
            %e_filer = sprintf('%s/%s.%s',new_file,models{1}{x}.img_id, params.datasets_params.file_ext);
            imshow(e_filer);
            %wpos = 100;
    
            disp('********* Start train linear SVM for exemplar  *********');
            
            %c_vals = [-4, -3, -2, 1, 2, 5, 10 ];
            %w1_vals = [20, 30, 50, 60, 80, 100, 150, 200, 400];
             c_vals = [-3];
             w1_vals = [60];

            for i = 1:length(c_vals)
                for j = 1:length(w1_vals)
                    
                fprintf(1,'strat to train with c=%d, w1=%d \n',c_vals(i), w1_vals(j));
                    
                weights = ones(size(train_features,1),1);
                weights(1) = w1_vals(j);

               % svm_model = fitcsvm(train_features, train_labels, 'PolynomialOrder', ...
               %                 [], 'BoxConstraint', 2^c_vals(i), 'KernelFunction', 'rbf', 'KernelScale','auto',...
               %                'Standardize', 1,'ClassNames', [-1; 1], 'Weights', weights);
                
                svm_model = fitcsvm(train_features, train_labels, 'KernelFunction', 'linear', 'PolynomialOrder', ...
                [], 'KernelScale', 1, 'BoxConstraint', 10^c_vals(i),...
               'Standardize', 1,'ClassNames', [-1; 1], 'Weights', weights);


                datas = cellfun(@(x)x.feature, train_datas{1},'UniformOutput',false);
                datas_1 = [vertcat(datas{:})];
                %normalize train_features
                %d_mean = mean(datas_1, 1);
                %d_std = std(datas_1, 1);
                %normalized_datas_1 = (datas_1 - repmat(d_mean, size(datas_1,1),1)) ./ repmat(d_std, size(datas_1,1),1);
                %fprintf(1,'dimension of wex is [%d %d]\n', size(wex,1),size(wex,2));
                %fprintf(1,'dimension of x is [%d %d]\n', size(m.x,1),size(m.x,2));
                %scores = wex*normalized_datas_1' - b;
                [predicted_labels, scores_both] = predict(svm_model, datas_1);
                predicted_scores = scores_both(:,2);
                num_postives = sum(predicted_scores>0);
                pos = predicted_scores(predicted_scores>0);
                fprintf(1,' --- Classify positives: %d\n',num_postives);
                %fprintf(1,'SVM iteration took %.3f sec, ',toc(starttime));
                
                [sorted_scores, indexes] = sort(predicted_scores, 'descend');
                %sorted_scores_1 = sorted_scores(1:12,:);
                %indexes_1 = sorted_scores(1:12,:);
                figure(2)
                
                for mm=1:3
                    for n=1:4
                        which = (mm-1)*4+n;
                        filer =train_datas{1}{indexes(which)}.img_filer;
                        img_temp = imread(filer);
                        img_temp = imresize(img_temp,[56 80]);
                        subplot(3,4,which); subimage(img_temp);
                        axis off;
                        title(sprintf('s=%f',sorted_scores(which)));
                    end
                end
                subtitle(sprintf('num positives=%d', num_postives));
                
                
                [neg_predicted_labels, neg_scores_both] = predict(svm_model, train_features(2:end,:));
                neg_predicted_scores = neg_scores_both(:,2);
                neg_num_postives = sum(neg_predicted_scores>0);
                neg_pos = neg_predicted_scores(neg_predicted_scores>0);
                fprintf(1,' --- Classify positives: %d\n',neg_num_postives);
                %fprintf(1,'SVM iteration took %.3f sec, ',toc(starttime));
                
                [neg_sorted_scores, neg_indexes] = sort(neg_predicted_scores,'ascend');
                %sorted_scores_1 = sorted_scores(1:12,:);
                %indexes_1 = sorted_scores(1:12,:);
                figure(3)
                
                for mm=1:3
                    for n=1:4
                        which = (mm-1)*4+n;
                        filer = m.neg_set.img_filers{neg_indexes(which)};
                        img_temp = imread(filer);
                        subplot(3,4,which); subimage(img_temp);
                        axis off;
                        title(sprintf('s=%f',neg_sorted_scores(which)));
                    end
                end
                subtitle(sprintf('neg num_positives=%d', neg_num_postives));
                
            end  
            end
        
            %% Linear SVM with 2 classes
%{         
pos_features_2 = cellfun(@(x)x.feature, train_datas{1,1}, 'UniformOutput', false);
pos_features_2 = [vertcat(pos_features_2{:})];
neg_features_2 = cellfun(@(x)x.feature, train_datas{1,2}, 'UniformOutput', false);
neg_features_2 = [vertcat(neg_features_2{:})];
pos_labels_2 = ones(size(pos_features_2,1),1);
neg_labels_2 = -ones(size(neg_features_2,1),1);
features_2 = vertcat(pos_features_2, neg_features_2);
labels_2 = vertcat(pos_labels_2, neg_labels_2);

            for i = 1:length(c_vals)
            
                    
                fprintf(1,'strat to train with c=%d\n',c_vals(i));


                svm_model_2 = fitcsvm(features_2, labels_2, 'KernelFunction', 'linear', 'PolynomialOrder', ...
                                [], 'KernelScale', 1, 'BoxConstraint', 10^c_vals(i), 'Standardize', 1,...
                                'ClassNames', [-1; 1]);


                datas = cellfun(@(x)x.feature, train_datas{1},'UniformOutput',false);
                datas_1 = [vertcat(datas{:})];
                %normalize train_features
                %d_mean = mean(datas_1, 1);
                %d_std = std(datas_1, 1);
                %normalized_datas_1 = (datas_1 - repmat(d_mean, size(datas_1,1),1)) ./ repmat(d_std, size(datas_1,1),1);
                %fprintf(1,'dimension of wex is [%d %d]\n', size(wex,1),size(wex,2));
                %fprintf(1,'dimension of x is [%d %d]\n', size(m.x,1),size(m.x,2));
                %scores = wex*normalized_datas_1' - b;
                [predicted_labels, scores_both] = predict(svm_model_2, datas_1);
                predicted_scores = scores_both(:,2);
                num_postives = sum(predicted_scores>0);
                pos = predicted_scores(predicted_scores>0);
                fprintf(1,' --- Classify positives: %d\n',num_postives);
                %fprintf(1,'SVM iteration took %.3f sec, ',toc(starttime));

            end

%}