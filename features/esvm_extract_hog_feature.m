function [feature, hog_size] = esvm_extract_hog_feature(img, params)
% Extract HoG feature of a given image.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

if ~exist('params', 'var');
    default_params = esvm_get_default_params;
    params = default_params.features_params.hog_params;
end

x = features_pedro(img, params.sbin);
hog_size = [size(x,1) size(x,2)];
feature = reshape(x,1,[]);

end