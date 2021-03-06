function [model, hn, mining_queue, mining_stats] = ...
    esvm_mine_negatives(model, mining_queue, imageset, mining_params)
% Compute detections "aka Hard-Negatives" hn for the images in the
% stream/queue [imageset/mining_queue] using current model
% 
% Input Data:
% model: current model in this iteration
% mining_queue: the mining queue create from
%    esvm_initialize_mining_queue(imageset)
% imageset: the source of images 
% mining_params: the parameters of the mining procedure
% 
% Returned Data: 
% hn: matrix which contains hard negatives mined using current model
%
% This file is modified based on Exemplar-SVM library.
% You can find its project here: https://github.com/quantombone/exemplarsvm
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

fprintf(1,'length of mining_queue is %d \n', length(mining_queue));

number_of_violating_images = 0;
violating_images = zeros(0,1);
empty_images = zeros(0,1);

numpassed = 0;

  if ~isfield(model,'total_mines')
    model.total_mines = 0;
  end

%if model does not have a field w, initialize this model
if ~isfield(model,'w')        
  model.w = model.x - mean(model.x(:));
  model.b = 0;
  model.svm_model.Mu = zeros(1,size(model.x,2));
  model.svm_model.Sigma = ones(1,size(model.x,2));
end

hn = zeros(size(model.x,2),0);

for i = 1:length(mining_queue)
  index = mining_queue{i}.index;
  x = load(imageset.feat_filers{index});
  x = x.data.feature;
  
  standarized_x = (x - model.svm_model.Mu) ./ model.svm_model.Sigma;
  s = model.w * standarized_x' + model.b;

  numpassed = numpassed + 1;

  mining_queue{i}.num_visited = mining_queue{i}.num_visited + 1;

  if s <= -1
    empty_images(end+1) = index;
  end
  
  %an image is violating if its prediction score bigger than -1,
  %else it is an empty image
  if s > -1
    if (mining_queue{i}.num_visited==1)
      number_of_violating_images = number_of_violating_images + 1;
    end 
    violating_images(end+1) = index;
    hn(:,end+1) = x';
  end
  
  if (numpassed + model.total_mines >= ...
      mining_params.train_max_mined_images) || ...
              (numpassed >= mining_params.train_max_images_per_iteration)

    fprintf(1,['Stopping mining and we have' ...
                                                ' %d new violators\n'],...
            number_of_violating_images);
    break;
  end
end

if isempty(hn) 
  mining_stats.num_violating = 0;
  mining_stats.num_empty = numpassed;
  mining_stats.total_mines = numpassed;
  fprintf(1,'No Violating images found for this iteration \n');
  model.no_negatives_found = true;
else

fprintf(1,'# Violating images: %d, #Non-violating images: %d\n', ...
        length(violating_images), length(empty_images));

mining_stats.num_empty = length(empty_images);
mining_stats.num_violating = length(violating_images);
mining_stats.total_mines = mining_stats.num_violating + mining_stats.num_empty;
end

%NOTE: there are several different mining scenarios possible here
%a.) dont process already processed images
%b.) place violating images at end of queue, eliminate free ones
%c.) place violating images at start of queue, eliminate free ones

if strcmp(mining_params.queue_mode,'onepass') == 1
  % MINING QUEUE UPDATE by removing already seen images
  mining_queue = update_mq_onepass(mining_queue, violating_images, ...
                                   empty_images);
elseif strcmp(mining_params.queue_mode,'cycle-violators') == 1
  % MINING QUEUE update by cycling violators to end of queue
  mining_queue = update_mq_cycle_violators(mining_queue, violating_images, ...
                                   empty_images);
elseif strcmp(mining_params.queue_mode,'front-violators') == 1
  % MINING QUEUE UPDATE by removing already seen images, and
  %front-placing violators (used in CVPR11)
  mining_queue = update_mq_front_violators(mining_queue, ...
                                           violating_images, ...
                                           empty_images);
else
  error(sprintf('Invalid queue mode: %s\n', ...
                mining_params.queue_mode));
end


function mining_queue = update_mq_onepass(mining_queue, violating_images, ...
                                           empty_images)

% Take the violating images and remove them from queue
mover_ids = find(cellfun(@(x)ismember(x.index,violating_images), ...
                         mining_queue));

mining_queue(mover_ids) = [];

% We now take the empty images and remove them from queue
mover_ids = find(cellfun(@(x)ismember(x.index,empty_images), ...
                         mining_queue));

mining_queue(mover_ids) = [];

function mining_queue = update_mq_front_violators(mining_queue,...
                                                  violating_images, ...
                                                  empty_images)
%An update procedure where the violating images are pushed to front
%of queue, and empty images are removed

% We now take the empty images and remove them from queue
mover_ids = find(cellfun(@(x)ismember(x.index,empty_images), ...
                         mining_queue));

%enders = mining_queue(mover_ids);
mining_queue(mover_ids) = [];


% We now take the violating images and place them on the start of the queue
mover_ids = find(cellfun(@(x)ismember((x.index),violating_images), ...
                         mining_queue));

starters = mining_queue(mover_ids);
mining_queue(mover_ids) = [];
mining_queue = cat(2,starters,mining_queue);


function mining_queue = update_mq_cycle_violators(mining_queue,...
                                                  violating_images, ...
                                                  empty_images)
%An update procedure where the violating images are pushed to front
%of queue, and empty images are pushed to back

% We now take the violating images and place them on the end of the queue
mover_ids = find(cellfun(@(x)ismember((x.index),violating_images), ...
                         mining_queue));

enders = mining_queue(mover_ids);
mining_queue(mover_ids) = [];
mining_queue = cat(2,mining_queue,enders);

% We now take the empty images and remove them from the queue
mover_ids = find(cellfun(@(x)ismember(x.index,empty_images), ...
                         mining_queue));

enders = mining_queue(mover_ids);
mining_queue(mover_ids) = [];
