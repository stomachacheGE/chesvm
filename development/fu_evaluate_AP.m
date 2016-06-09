function fu_evaluate_AP(labels_test, prob_estim, split_data)

result_dir = './results';

if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

labels = unique(labels_test);


for i = 1: length(labels)
    %initialize ground-truth pairs with -1
    ids = 1: length(labels_test);
    gt = ones(1, length(labels_test)) * -1;
    %set the corresponding gt to 1
    idxs = find(labels_test==labels(i));
    gt(idxs) = 1;
    
    %store the gt .txt file to disk
    gt_filer = sprintf('%s/%s.txt',...
                result_dir,split_data(i).classname);
    fprintf(1,'Writing %s Classification Ground-truth File %s\n',split_data(i).classname, gt_filer);
    fid = fopen(gt_filer,'w');
    for j = 1:length(ids)
      fprintf(fid,'%d %d \n',ids(j), gt(j));
    end
    fclose(fid);
    
    %store the classification results file to disk
    res_filer = sprintf('%s/%s_res.txt',...
                result_dir,split_data(i).classname);
    fprintf(1,'Writing %s Classification Results File %s\n',split_data(i).classname, res_filer);
    fid2 = fopen(res_filer,'w');
    for j = 1:length(ids)
      fprintf(fid2,'%d %f \n',ids(j), prob_estim(j,i));
    end
    fclose(fid2);
    
    VOCevalcls(split_data(i).classname, result_dir, 'true');
    
end

end

function [rec,prec,ap] = VOCevalcls(cls,result_dir,draw)

% load test set

gt_filer = sprintf('%s/%s.txt',result_dir,cls);
[gtids,gt]=textread(gt_filer,'%s %d');

% load results
res_filer = sprintf('%s/%s_res.txt',result_dir,cls);
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

