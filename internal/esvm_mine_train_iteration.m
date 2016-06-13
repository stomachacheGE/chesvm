function [m] = esvm_mine_train_iteration(m, feat_name, training_function)
% ONE ITERATION OF: Mine negatives until cache is full and update the current
% classifier using training_function (do_svm, do_rank, ...). m must
% contain the field m.neg_set, which indicates the current
% training set of negative images
% Returns the updated model (where m.mining_queue is updated mining_queue)
%
% This file is modified based on Exemplar-SVM library.
% You can find its project here: https://github.com/quantombone/exemplarsvm

% Start wtrace (trace of learned classifier parameters across
% iterations) with first round classifier, if not present already

if ~isfield(m,'wtrace')
  m.wtrace{1} = m.x - mean(m.x(:));
  m.btrace{1} = 0;
end

if length(m.mining_queue) == 0
  fprintf(1,' ---Null mining queue, not mining!\n');
  return;
end

[hn, m.mining_queue, mining_stats] = ...
      esvm_mine_negatives(m, m.mining_queue, m.neg_set, ...
                     feat_name, m.mining_params);


if ~isempty(hn)
    m = add_new_detections(m, hn);
end
               
m = update_the_model(m, mining_stats, training_function);


end

function [m] = update_the_model(m, mining_stats, training_function)
%% UPDATE the current SVM, keep max number of svs, and show the results

if ~isfield(m,'mining_stats')
  m.mining_stats{1} = mining_stats;
else
  m.mining_stats{end+1} = mining_stats;
end

m = training_function(m);

% Append new w to trace
m.wtrace{end+1} = m.w;
m.btrace{end+1} = m.b;
end

function m = add_new_detections(m, hn)
% Add detected hard-negatives to support vectors m.svxs
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

%First iteration might not have support vector information stored
if ~isfield(m, 'svxs') || isempty(m.svxs)
  m.svxs = [];
end

m.svxs = cat(2,m.svxs,hn);
end


