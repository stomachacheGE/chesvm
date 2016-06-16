
m.neg_set = neg_set{1};
m.mining_queue = esvm_initialize_mining_queue(m.neg_set);
scores = zeros(length(m.mining_queue),1);
for i = 1:length(m.mining_queue)
  index = m.mining_queue{i}.index;
  x = load(m.neg_set.feat_filers{index});
  x = x.data.feature;
  model = m;
  level = 5;
  standarized_x = (x - model.mutrace{level}) ./ model.sigmatrace{level};
  scores(i) = model.wtrace{level} * standarized_x' + model.btrace{level};
  %scores(i) = model.w * standarized_x' + model.b;
  %[~, s] = predict(m.svm_model, x);
  %scores(i) = s(2);
end

 [sorted,idxs] = sort(scores,'descend');
%% 
m.neg_set = neg_set{1};
m.mining_queue = esvm_initialize_mining_queue(m.neg_set);
neg = 100;
neg_feat = zeros(neg,size(m.x,2));
neg_feat(1,:) = m.x;
for i = 1:neg
  index = m.mining_queue{i}.index;
  x = load(m.neg_set.feat_filers{index});
  x = x.data.feature;
  neg_feat(i+1,:) = x;
  %scores(i) = model.w * standarized_x' + model.b;
  %[~, s] = predict(svm_model, x);
  %scores(i) = s(2);
end

 superx = neg_feat;
 supery = -1*(ones(neg+1,1));
 supery(1) = 1;
 weights = ones(size(superx,1),1);
 weights(1) = 60;
 c = -3;
 svm_model = fitcsvm(superx, supery,'PolynomialOrder', ...
                            [],'BoxConstraint', 2^c, 'KernelFunction', 'linear', 'KernelScale',1,...
                           'Standardize', 1,'ClassNames', [-1; 1], 'Weights', weights);
                       
 svm_weights = full(sum((svm_model.SupportVectors)  .* ...
                                         repmat((svm_model.Alpha.* svm_model.SupportVectorLabels),1, ...
                                                size(svm_model.SupportVectors,2)),1));
 b = svm_model.Bias;
 
 for i = 1:length(m.mining_queue)
  index = m.mining_queue{i}.index;
  x = load(m.neg_set.feat_filers{index});
  x = x.data.feature;
  model = svm_model;
  standarized_x = (x - model.Mu) ./ model.Sigma;
  %scores(i) = model.wtrace{6} * standarized_x' + model.btrace{6};
  scores(i) = svm_weights * standarized_x' + b;
  %[~, s] = predict(svm_model, x);
  %scores(i) = s(2);
 end

 [sorted,idxs] = sort(scores,'descend');