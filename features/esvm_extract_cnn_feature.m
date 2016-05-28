function feature = esvm_extract_cnn_feature(img, convnet, layer)

   normfeat = 1; % normalize feature after extraction?
   

    [~, idx] = find(cellfun(@(x) any(strcmp(x.name,layer)),convnet.layers));
    
    
    convnet.layers = convnet.layers(1:idx); % largest output layer id is the last layer in the network
 
    % Information about avarage CNN image

    meanim = convnet.meta.normalization.averageImage;
    if size(meanim,2) == 1 % gs Is a vector? Make image from the mean vector
        meanim = reshape(meanim, 1,1,3);
        meanim = repmat(meanim, convnet.meta.normalization.imageSize(1:2));
    end
    
    imageSize = convnet.meta.normalization.imageSize;
    I = imresize(single(img),imageSize(1:2)); % read and convert/resize to the CNN image sizes
    
    if size(I,3) == 1 % if we have a grayscale image
        I = I(:,:,[1 1 1]); % represent represent it as RGB
    end
    
    I = I - meanim; % subtract from average image               

    feature = vl_simplenn(convnet, I, [], []); % perform feature extraction
    feature = squeeze(gather(feature(idx(1)+1).x))';  %fu. why only use the first last_cnn_name layer output?

    if normfeat
        feature = feature./max(feature);
    %fu. from paper "Return of the Devil in the Details:Delving Deep
    %fu. into Convolutional Nets", linear SVM benefit from L2 norm?
    %fu. quote:"an interpretation is
    %that after normalisation the inner product corresponds to the
    %cosine similarly"|

    % normalize (L1 norm)
    % cnn_res = cnn_res./sum(cnn_res);            
    end         
        
end

