function feature = esvm_extract_cnn_feature(img, convnet, layer)
% Extract CNN feature of a given image.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

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
feature = squeeze(gather(feature(idx(1)+1).x))';  

if normfeat
    feature = feature./max(feature);
%from paper "Return of the Devil in the Details:Delving Deep
%into Convolutional Nets", linear SVM benefit from L2 norm?
%quote:"an interpretation is
%that after normalisation the inner product corresponds to the
%cosine similarly"|          
end         
        
end

