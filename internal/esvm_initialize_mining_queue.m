function mining_queue = esvm_initialize_mining_queue(imageset, ordering)
%Initialize the mining queue with ordering (random by default)

% This file is modified based on Exemplar-SVM library.
% You can find its project here: https://github.com/quantombone/exemplarsvm
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

if ~exist('ordering','var')
  fprintf(1,'Randomizing mining queue\n');
  myRandomize;
  ordering = randperm(length(imageset.feat_filers));
end

mining_queue = cell(0,1);
for zzz = 1:length(ordering)
  mining_queue{zzz}.index = ordering(zzz);
  mining_queue{zzz}.num_visited = 0;
end

