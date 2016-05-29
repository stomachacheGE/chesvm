function [models, cal_sets, neg_sets] = esvm_train_initialization(datas, feat_name)

    models = cell(1,length(datas));
    tmp_neg_sets = cell(1,length(datas));
    neg_sets = cell(1,length(datas));
    cal_sets = cell(1,length(datas));
    
    fprintf(1, 'Initializing models and negatives ... \n');
    
    for i=1:length(datas)
        left = 1:length(datas);
        left = left(left~=i);
        not_that_class = datas(1,left);
        
        feat_filers = cell(1, length(datas)-1);
        img_filers = cell(1, length(datas)-1);
        img_sizes = cell(1, length(datas)-1);
        
        for j=1:length(datas)-1
            feat_filers_temp = cellfun(@(x) x.feat_filer, not_that_class{j}, 'UniformOutput', false);
            %feat_filers{j} = [horzcat(feat_filers_temp{:})];
            feat_filers{j} = feat_filers_temp;
            img_filers_temp = cellfun(@(x) x.img_filer, not_that_class{j}, 'UniformOutput', false);
            %img_filers{j} = [horzcat(img_filers_temp{:})];  
            img_filers{j} = img_filers_temp;
            img_sizes_temp = cellfun(@(x) x.img_size, not_that_class{j}, 'UniformOutput', false);
            %img_filers{j} = [horzcat(img_filers_temp{:})];  
            img_sizes{j} = img_sizes_temp;
        end    
        tmp_neg_sets{i}.feat_filers = [horzcat(feat_filers{:})];
        tmp_neg_sets{i}.img_filers = [horzcat(img_filers{:})];  
        tmp_neg_sets{i}.img_sizes = [horzcat(img_sizes{:})];  
    end
    
    for i=1:length(datas)
        
        models{i} = cell(1,length(datas{i}));
        neg_sets{i} = cell(1,length(datas{i}));
        cal_sets{i} = cell(1,length(datas{i}));
        
        for j=1:length(datas{i})
            
            data = datas{i}{j};
            feature = data.feature;
            
            clear model
            %model.w = feature - mean(feature(:));
            %model.b = 0;
            model.x = feature;
            model.label = data.label;
            model.cls_name = data.cls_name;
            model.img_id = data.img_id;
            model.img_size = data.img_size;
      
            if strcmp(feat_name,'hog')
                 model.hog_size = data.hog_size;
            end
            models{i}{j} = model;
            
            myRandomize;
            ordering = randperm(length(tmp_neg_sets{i}.img_filers));
            %take 1/10 of tmo_neg_sets for 
            num_neg = int16(length(tmp_neg_sets{i}.img_filers) / 20);
            neg_for_cal = tmp_neg_sets{i}.feat_filers(ordering(1:num_neg));
            pos_for_cal = cellfun(@(x)x.feat_filer, datas{i}, 'UniformOutput', false);
            %pos_for_cal = [horzcat(pos_for_cal{:})];
            cal_sets{i}{j}.feat_filers = cat(2, pos_for_cal, neg_for_cal);
            neg_img_for_cal = tmp_neg_sets{i}.img_filers(ordering(1:num_neg));
            pos_img_for_cal = cellfun(@(x)x.img_filer, datas{i}, 'UniformOutput', false);
            cal_sets{i}{j}.img_filers = cat(2, pos_img_for_cal, neg_img_for_cal);
            cal_sets{i}{j}.num_neg = num_neg;
            %cal_sets{i}{j}.gt = [ones(1, length(pos_for_cal)) zeros(1,length(neg_for_cal))];
            
            neg_sets{i}{j}.feat_filers = tmp_neg_sets{i}.feat_filers(ordering(num_neg+1:end));
            neg_sets{i}{j}.img_filers = tmp_neg_sets{i}.img_filers(ordering(num_neg+1:end));
            neg_sets{i}{j}.img_sizes = tmp_neg_sets{i}.img_sizes(ordering(num_neg+1:end));
        end
    end
    
    
    fprintf(1, 'Initializing models and negatives finished. \n');
end