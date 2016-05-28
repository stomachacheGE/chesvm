function ap_res = esvm_evaluate_AP(predictions, test_datas, use_algorithm, use_feature, params)

    datasets_params = params.datasets_params;

    classifi_res_dir = fullfile('.', datasets_params.results_folder,'classifications');
    ground_truth_dir = fullfile(classifi_res_dir, 'ground_truth');
    lin_svm_res_dir = fullfile(classifi_res_dir, 'linsvm');
    esvm_res_dir = fullfile(classifi_res_dir, 'esvm');
    
    if ~exist(classifi_res_dir, 'dir')
        mkdir(classifi_res_dir)
    end

    if ~exist(ground_truth_dir, 'dir')

        %mkdir(classifi_res_dir);
        mkdir(ground_truth_dir);
        mkdir(lin_svm_res_dir);
        mkdir(esvm_res_dir);

        labels = unique(cellfun(@(x) x{1}.label, test_datas));

        fprintf(1,'Writing Classification Ground-truth Files.\n');
       %% create ground_truth .txt files
        %initialize ground-truth pairs with -1
        for i =1: length(labels);

            %find the corresponding class name of this label
            idx = 1;
            while(idx <= length(test_datas))
                if test_datas{idx}{1}.label == labels(i)
                    break;
                else
                    idx = idx + 1;
                end
            end

            cls_name = test_datas{idx}{1}.cls_name;
            %store the gt .txt file to disk
            gt_filer = sprintf('%s/%s_gt.txt',...
                        ground_truth_dir,cls_name);
            
            fid = fopen(gt_filer,'w');

            for mm = 1:length(test_datas)
                for j = 1:length(test_datas{mm})
                  if test_datas{mm}{j}.label == i
                      fprintf(fid,'%s 1 \n',test_datas{mm}{j}.img_id);
                  else
                      fprintf(fid,'%s -1 \n',test_datas{mm}{j}.img_id);
                  end
                end
            end
            fclose(fid);      
        end
    end

    if strcmp(use_algorithm, 'svm')
        ap_res_filer = [lin_svm_res_dir '/' use_feature '_' use_algorithm '.txt'];
    else
        ap_res_filer = [esvm_res_dir '/' use_feature '_' use_algorithm '.txt'];
    end
    

       
    
    %if ap result file does not exist, evaluate and store the results
    if ~exist(ap_res_filer,'file')
        
        labels_test = unique(predictions.ids);
        %store the classification results file to disk
        ap_res = cell(1, length(labels_test));
        
        fprintf(1,'Writing Classification Results To File.\n');
        for q=1:length(labels_test)

            %find the corresponding class name of this label
            clear idx;

            idx = 1;
            while(idx <= length(test_datas))
                if test_datas{idx}{1}.label == labels_test(q)
                    break;
                else
                    idx = idx + 1;
                end
            end

            cls_name = test_datas{idx}{1}.cls_name;

            if strcmp(use_algorithm, 'svm')       
                res_filer = sprintf('%s/%s_%s_res.txt',...
                        lin_svm_res_dir,cls_name, use_feature);
            else
                res_filer = sprintf('%s/%s_%s_res.txt',...
                        esvm_res_dir,cls_name, use_feature);
            end

            if ~exist(res_filer,'file')

                fprintf(1,'Writing %s Classification Results File %s\n',upper(cls_name), res_filer);
                fid2 = fopen(res_filer,'w');
                counter = 1;
                for mm = 1:length(test_datas)
                     for m = 1:length(test_datas{mm})
                       
                         fprintf(fid2,'%s %f \n',test_datas{mm}{m}.img_id, predictions.prob(counter,q));
                         counter = counter + 1;
                     end
                end
                fclose(fid2);            
            end



            if strcmp(use_algorithm, 'svm')       
                [~,~,ap_res{q}.ap] = VOCevalcls(cls_name, use_feature, lin_svm_res_dir, ground_truth_dir, 'true');       
            else
                [~,~,ap_res{q}.ap] = VOCevalcls(cls_name, use_feature, esvm_res_dir, ground_truth_dir, 'true');
            end
            ap_res{q}.cls_name = cls_name;
                  
        end
        
        fprintf(1,'Writing Average Precision Results To File: %s\n',ap_res_filer);
            fid3 = fopen(ap_res_filer,'w');
            for m = 1:length(ap_res)
               fprintf(fid3,'%s %f \n',ap_res{m}.cls_name, ap_res{m}.ap);
            end
        fclose(fid3);  
        
    else
        
        fprintf(1,'Loading Average Precision Results From File: %s\n',ap_res_filer);
        [temp.cls temp.ap] = textread(ap_res_filer, '%s %f');
        clear q;
        for q = 1:length(temp.cls)
            ap_res{q}.cls_name = temp.cls{q};
            ap_res{q}.ap = temp.ap(q);

            cls_name = temp.cls{q};
            if strcmp(use_algorithm, 'svm')       
                [~,~,~] = VOCevalcls(cls_name, use_feature, lin_svm_res_dir, ground_truth_dir, 'true');       
            else
                [~,~,~] = VOCevalcls(cls_name, use_feature, esvm_res_dir, ground_truth_dir, 'true');
            end
        end
    end

end

function [rec,prec,ap] = VOCevalcls(cls, feat_name, result_dir, ground_truth_dir,draw)

% load test set

gt_filer = sprintf('%s/%s_gt.txt',ground_truth_dir,cls);
[gtids,gt]=textread(gt_filer,'%s %d');

% load results
res_filer = sprintf('%s/%s_%s_res.txt',result_dir,cls,feat_name);
[ids,confidence]=textread(res_filer,'%s %f');

% map results to ground truth images
out=ones(size(gt))*-inf;

tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: pr: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    % find ground truth image
    j=strmatch(ids{i},gtids,'exact');
    if isempty(j)
        error('unrecognized image "%s"',ids{i});
    elseif length(j)>1
        error('multiple image "%s"',ids{i});
    else
        out(j)=confidence(i);
    end
end
% compute precision/recall

[so,si]=sort(-out);
tp=gt(si)>0;
fp=gt(si)<0;

fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/sum(gt>0);
prec=tp./(fp+tp);

ap=VOCap(rec,prec);

if draw
    % plot precision/recall
    figure('Name',cls);
    plot(rec,prec,'-');
    grid;
    xlim([0 1])
    ylim([0 1])
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, AP = %.3f',cls,ap));
end

end

function ap = VOCap(rec,prec)

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));

end





