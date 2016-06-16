function [m,other] = esvm_update_svm(m)
% Perform SVM learning for a single exemplar model, we assume that
% the exemplar has a set of detections loaded in m.svxs 
% Durning Learning, we can apply some pre-processing such as PCA or
% dominant gradient projection
%
% This file is modified based on Exemplar-SVM library.
% You can find its project here: https://github.com/quantombone/exemplarsvm

other = 'svm';
%if no inputs are specified, just return the suffix of current method
if nargin==0
  m = '-svm';
  return;
end

mining_params = m.mining_params;
xs = m.svxs;

%NOTE: MAXSIZE should perhaps be inside of the default_params script?
%{
    MAXSIZE = 3500;
    if size(xs,2) >= MAXSIZE
      HALFSIZE = MAXSIZE/2;
      %NOTE: random is better than top 5000
      r = m.w*xs;
      [tmp,r] = sort(r,'descend');
      r1 = r(1:HALFSIZE);

      r = HALFSIZE+randperm(length(r((HALFSIZE+1):end)));
      r = r(1:HALFSIZE);
      r = [r1 r];
      xs = xs(:,r);
    end

%}
  
superx = cat(2,m.x',xs)';
supery = cat(1,ones(size(m.x',2),1),-1*ones(size(xs,2),1));
spos = sum(supery==1);
sneg = sum(supery==-1);

%wpos = mining_params.train_positives_constant;
%wneg = 1;

fprintf(1,' -----\nStarting SVM: dim=%d... #pos=%d, #neg=%d ',...
        size(superx,2),spos,sneg);
starttime = tic;

%svm_model = svmtrain(supery, superx',sprintf(['-s 0 -t 0 -c' ...
%                    ' %f -w1 %.9f -q'], mining_params.train_svm_c, wpos));
 weights = ones(size(superx,1),1);
 weights(1) = mining_params.train_positives_constant;
 c = mining_params.train_svm_c;
 m.svm_model = fitcsvm(superx, supery,'PolynomialOrder', ...
                            [],'BoxConstraint', 2^c, 'KernelFunction', 'linear', 'KernelScale',1,...
                           'Standardize', 1,'ClassNames', [-1; 1], 'Weights', weights);
                     
if size(m.svm_model.SupportVectors,1) == 0
  %learning had no negatives
  wex = m.w;
  b = m.b;
  fprintf(1,'reverting to old model...\n');
else

  %convert support vectors to decision boundary
  %svm_weights = full(sum(svm_model.SVs .* ...
  %                       repmat(svm_model.sv_coef,1, ...
  %                               size(svm_model.SVs,2)),1));
  
  svm_weights = full(sum((m.svm_model.SupportVectors)  .* ...
                                         repmat((m.svm_model.Alpha.* m.svm_model.SupportVectorLabels),1, ...
                                                size(m.svm_model.SupportVectors,2)),1));
  b = m.svm_model.Bias;
  wex = svm_weights;
    
  if supery(1) == -1
    wex = wex*-1;
    b = b*-1;    
  end

  %% issue a warning if the norm is very small
  if norm(wex) < .00001
    fprintf(1,'learning broke down!\n');
  end  
end

%fprintf(1,'dimension of wex is [%d %d]\n', size(wex,1),size(wex,2));
%fprintf(1,'dimension of x is [%d %d]\n', size(m.x,1),size(m.x,2));
%maxpos = max(wex*m.x' - b);
standarized_x = (m.x - m.svm_model.Mu) ./ m.svm_model.Sigma;
%replace any entry which is finite with 0
[row, col] = find(~isfinite(standarized_x));
standarized_x(row,col) = 0;
    
maxpos = standarized_x * svm_weights' + b;
fprintf(1,' --- Max positive is %.3f\n',maxpos);
fprintf(1,'SVM iteration took %.3f sec, ',toc(starttime));

m.w = wex;
m.b = b;

repmat_Mu = repmat(m.svm_model.Mu',1,size(m.svxs,2));
repmat_Sigma = repmat(m.svm_model.Sigma',1, size(m.svxs,2));
standarized_svxs = (m.svxs - repmat_Mu) ./ repmat_Sigma;
%replace any entry which is finite with 0
[row, col] = find(~isfinite(standarized_svxs));
standarized_svxs(row,col) = 0;
    
r = m.w*standarized_svxs + m.b;
svs = find(r >= -1.0000);

if length(svs) == 0
  fprintf(1,' Note - number of negative support vectors is 0! \n');
  m.no_negatives_found = true;
else
  m.svxs = m.svxs(:,svs);
end
%m.svbbs = m.svbbs(svs,:);
%fprintf(1,' kept %d negatives\n',total_length);
fprintf(1,' kept %d negatives\n',length(svs));

