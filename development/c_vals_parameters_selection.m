          qq = 1;
%          fprintf(1,'Strat to train model using linear svm %d/%d, model_id = %s, class = %s \n', counter, ...
        %                                num_models,m.img_id, m.cls_name);
          % Add training set and training set's mining queue 
          %fprintf(1,'q is %d\n',qq);
          %fprintf(1,'neg_set size is [%d %d]\n', size(neg_set,1),size(neg_set,2));
          m = models{1}{1};
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
            m = esvm_mine_train_iteration(m, feat_name, m.mining_params.training_function);

            if ((total_mines >= params.training_params.train_max_mined_images) || ...
                  (isempty(m.mining_queue))) || ...
                  (m.iteration == params.training_params.train_max_mine_iterations)

              keep_going = 0;      
              %bump up filename to final file
              filer2 = filer2final;
            end
          %}
            feat_name = 'cnn';
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
                 [temp, ~] = params.training_params.hog_extractor(img);
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
            
            %m = linSVM_train_exemplar(m, train_features, train_labels, params);
            
            %normalize train_features
            t_mean = mean(train_features, 1);
            t_std = std(train_features, 1);
            normalized_train_features = (train_features - repmat(t_mean, size(train_features,1),1)) ./ repmat(t_std, size(train_features,1),1);
            
            
            wpos = 100;
    
            disp('********* Start train linear SVM for exemplar  *********');
            
            c_vals = [-5, -4, -2, 1, 2, 5, 10 ];
            w1_vals = [100000, 1500, 2000,1068];
            
            for i = 1:length(c_vals)
                for j = 1:length(w1_vals)
                    
                fprintf(1,'strat to train with c=%d, w1=%d \n',c_vals(i), w1_vals(j));
                svm_model = libsvmtrain(train_labels,normalized_train_features, sprintf(('-s 0 -t 0 -c %f -w1 %.9f'),...
                                                             10^c_vals(i), w1_vals(j)));
                                        

                    if length(svm_model.sv_coef) == 0
                      %learning had no negatives
                      wex = m.w;
                      b = m.b;
                      fprintf(1,'no supporting vectors...\n');
                    else

                      %convert support vectors to decision boundary
                      svm_weights = full(sum(svm_model.SVs .* ...
                                             repmat(svm_model.sv_coef,1, ...
                                                    size(svm_model.SVs,2)),1));

                      wex = svm_weights;
                      b = svm_model.rho;

                      %{
                      %% project back to original space
                      b = b + wex'*A(m.mask,:)'*mu(m.mask);
                      wex = A(m.mask,:)*wex;

                      wex2 = zeros(size(superx,1),1);
                      wex2(m.mask) = wex;

                      wex = wex2;
                      %}
                      %% issue a warning if the norm is very small
                      if norm(wex) < .00001
                        fprintf(1,'learning broke down!\n');
                      end  
                    end

                    datas = cellfun(@(x)x.feature, train_datas{1},'UniformOutput',false);
                    datas_1 = [vertcat(datas{:})];
                    %normalize train_features
                    d_mean = mean(datas_1, 1);
                    d_std = std(datas_1, 1);
                    normalized_datas_1 = (datas_1 - repmat(d_mean, size(datas_1,1),1)) ./ repmat(d_std, size(datas_1,1),1);
                    %fprintf(1,'dimension of wex is [%d %d]\n', size(wex,1),size(wex,2));
                    %fprintf(1,'dimension of x is [%d %d]\n', size(m.x,1),size(m.x,2));
                    scores = wex*normalized_datas_1' - b;
                    num_postives = sum(scores>0);
                    pos = scores(scores>0);
                    fprintf(1,' --- Classify positives: %d\n',num_postives);
                    %fprintf(1,'SVM iteration took %.3f sec, ',toc(starttime));

                end
            end
