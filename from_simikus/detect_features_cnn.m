function detect_features_cnn(im_dir,file_ext,varargin)
   normfeat = 1; % normalize feature after extraction?
   
    if nargin > 2
        % signal image file extension
        img_ext = cell2mat(varargin(1));         %fu. file_ext:jpg
        % desired output layer name of CNN model
        last_cnn_name = cell2mat(varargin(2)); %fu. out_cnn_name:relu6
        cnn_path = cell2mat(varargin(3));  %fu. :/train
    end      
        
    % do explicitely check whether im_dir is really a folder or
    % not; if not, Matlab 2014 (on workstation) behaves differently than
    % Matlab 2013 which is the reason for this workaround
    if isdir(im_dir)
        dd = dir(fullfile(im_dir, ['*.' img_ext]));
    else
        dd = [];
    end
    
    % Load pretrained CNN model
    convnet = load(cnn_path);   
    
    % Change output structure 
    if ischar(last_cnn_name)
        out_cnn_name = last_cnn_name;
        last_cnn_name = cellstr(last_cnn_name); 
    end % make sure we have a cell
    if any(strcmp(last_cnn_name,'prob')) % if any layer is a prob, replace by relu
        convnet.layers{end} = struct('name', 'relu8', ...
            'type', 'relu', ...
            'leak', 0, ...
            'precious', 0) ;
        last_cnn_name{find(strcmp(last_cnn_name,'prob'))} = 'relu8'; % replace 'prob' with relu
    end

    % find index of desired output layer with given layer type name
    lidx = zeros(1,numel(last_cnn_name)); % how many results we will need
    for i = 1:numel(last_cnn_name)
        [~, lidx(i)] = find(cellfun(@(x) any(strcmp(x.name,last_cnn_name{i})),convnet.layers));
        %fu. [~, lidx(i)] = find(cellfun(@(x) strcmp(x.name,last_cnn_name{i}),convnet.layers));
    end
    convnet.layers = convnet.layers(1:max(lidx)); % largest output layer id is the last layer in the network
 
    % Information about avarage CNN image
    imageSize = convnet.meta.normalization.imageSize;
    if length(imageSize)<4 % make sure it is 3d dim
        imageSize = [imageSize, 1];
    end  
    meanim = convnet.meta.normalization.averageImage;
    if size(meanim,2) == 1 % gs Is a vector? Make image from the mean vector
        meanim = reshape(meanim, 1,1,3);
        meanim = repmat(meanim, convnet.meta.normalization.imageSize(1:2));
    end      
    
    for i = 1:length(dd)
%     parfor i = 1:length(dd)
        fname = fullfile(im_dir,dd(i).name);
        dir_out = fullfile(im_dir, ['feats_' out_cnn_name]);
        if ~exist(dir_out, 'dir')
            mkdir(dir_out)
        end
        fname_out = fullfile(            fprintf('feature dimension: %s', size(feature.feature));dir_out,strcat(dd(i).name(1:end-3),file_ext)); % 
        
        if exist(fname_out,'file') %fu. (comment out since rwfeat is not defined) && ~rwfeat 
            %if verbose, 
            fprintf('File exists! Skipping %s \n',fname_out); 
            %end
            continue;
        end
        
        fprintf('Extracting features with pretrained CNN model: %s \n',fname_out);
        I = imresize(single(imread(fname)),imageSize(1:2)); % read and convert/resize to the CNN image sizes
        if size(I,3) == 1 % if we have a grayscale image
            I = I(:,:,[1 1 1]); % represent represent it as RGB
        end
        I = I - meanim; % subtract from average image               
        
        cnn_res = vl_simplenn(convnet, I, [], []); % perform feature extraction
        cnn_res = squeeze(gather(cnn_res(lidx(1)+1).x))';  %fu. why only use the first last_cnn_name layer output?
        
        if normfeat
            cnn_res = cnn_res./max(cnn_res);
        %fu. from paper "Return of the Devil in the Details:Delving Deep
        %fu. into Convolutional Nets", linear SVM benefit from L2 norm?
        %fu. quote:"an interpretation is
        %that after normalisation the inner product corresponds to the
        %cosine similarly"|

        % normalize (L1 norm)
        % cnn_res = cnn_res./sum(cnn_res);            
        end         
        
        desc = struct('cnnfeature',cnn_res);
        
        % clear rad in order to avoid problems when assigning values
        % since its type changes between cell and double array
        cnn_res = [];
        
        iSave(desc,fname_out);
    end

end


function iSave(desc,fName)
    save(fName,'desc');
end

