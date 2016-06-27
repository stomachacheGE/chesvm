function results = esvm_evaluate_ACC(prediction, test_datas, use_algorithm, use_feature, cal, hard_negative, params)

    datasets_params = params.datasets_params;

    classifi_res_dir = fullfile('.', datasets_params.results_folder,'classifications');
    ground_truth_dir = fullfile(classifi_res_dir, 'ground_truth');
    lin_svm_root_res_dir = fullfile(classifi_res_dir, 'linsvm');
    esvm_res_root_dir = fullfile(classifi_res_dir, 'esvm');

    
    if ~exist(classifi_res_dir, 'dir')
        mkdir(classifi_res_dir)
    end

    if ~exist(ground_truth_dir, 'dir')
        mkdir(ground_truth_dir);
    end
    
    if ~exist(lin_svm_root_res_dir,'dir')
        mkdir(lin_svm_root_res_dir);
    end

    if hard_negative
        esvm_res_hn_dir = fullfile(esvm_res_root_dir, 'hard_negative');
    else
        esvm_res_hn_dir = fullfile(esvm_res_root_dir, 'wo_hard_negative');
    end
    
    if cal
        esvm_res_cal_dir = fullfile(esvm_res_hn_dir, 'calibration');
    else
        esvm_res_cal_dir = fullfile(esvm_res_hn_dir, 'wo_calibration');
    end
    
    esvm_res_dir = fullfile(esvm_res_cal_dir, use_feature);
    lin_svm_res_dir = fullfile(lin_svm_root_res_dir, use_feature);
    
    if strcmp(use_algorithm,'esvm')
        if ~exist(esvm_res_hn_dir,'dir')
            mkdir(esvm_res_hn_dir);
        end

        if ~exist(esvm_res_cal_dir,'dir')
            mkdir(esvm_res_cal_dir);
        end

        if ~exist(esvm_res_dir,'dir')
            mkdir(esvm_res_dir);
        end
    
    else
        if ~exist(lin_svm_res_dir,'dir')
            mkdir(lin_svm_res_dir);
        end
    end
    classes = cellfun(@(x) x{1}.cls_name, test_datas,'UniformOutput', false)


   %% create ground_truth 
        all_gt = cell(1, length(test_datas));
        for mm = 1:length(test_datas)            
            all_gt{mm} = cellfun(@(x) x.label, test_datas{mm});
        end
        all_gt = horzcat(all_gt{:})';

    
    if strcmp(use_algorithm, 'svm')
        acc_filer = [lin_svm_res_dir '/' use_feature '_' use_algorithm '_acc.txt'];
    else
        acc_filer = [esvm_res_dir '/' use_feature '_' use_algorithm '_acc.txt'];
    end     

    % caculate accurracies
    res = classperf(all_gt, prediction.ids);
    results.accuraccy = res.CorrectRate;
    results.acc_per_cls = 1 - res.ErrorDistributionByClass./res.SampleDistributionByClass;
    results.classes = classes;
    results.conf_mat = confusionmat(all_gt, prediction.ids);
    results.mean_accuraccy = mean(res.CorrectRate);
    heatmap(results.conf_mat, classes, classes, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
    %if ap result file does not exist, evaluate and store the results
    if ~exist(acc_filer,'file')
        fid = fopen(acc_filer,'w');

        for i = 1:length(results.classes)

            fprintf(fid,'%s %f \n', classes{i}, results.acc_per_cls(i));
        end
        fclose(fid);   
    end
        
    num_imgs = cellfun(@(x) length(x), test_datas, 'UniformOutput', false);
    num_imgs = cell2mat(num_imgs);

    offset(1) = 0;
    for i=1:length(num_imgs)
        offset(i+1) = sum(num_imgs(1:i));
    end
    
   offset
    mis = cell(1,0);
    for i = 1:length(all_gt)
        if all_gt(i)~= prediction.ids(i)
            minus = i - offset;
            idxes = find(minus>0);
            ii = idxes(end);
            jj =  i - offset(ii);
            mis(end+1) = {test_datas{ii}{jj}.img_id};
        end
    end
    results.misclassifications = mis;   
    
    if strcmp(use_algorithm, 'svm')
        mis_filer = [lin_svm_res_dir '/' use_feature '_' use_algorithm '_misclassifications.txt'];
    else
        mis_filer = [esvm_res_dir '/' use_feature '_' use_algorithm '_misclassifications.txt'];
    end 
    mis_filer
    if ~exist(mis_filer,'file')
        fid = fopen(mis_filer,'w');

        for i = 1:length(mis)

            fprintf(fid,'%s \n', mis{i});
        end
        fclose(fid);   
    end

end

