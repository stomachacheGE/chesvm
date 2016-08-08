% Use this script to train exemplar-SVMs and linear SVMs with all available
% algorithms and features.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

esvm_train('hog','svm', false, true);
esvm_train('hog','esvm',false,false);
esvm_train('hog','esvm',false,true);
esvm_train('hog','esvm',true,false);
esvm_train('hog','esvm',true,true);

esvm_train('cnn','svm', false, true);
esvm_train('cnn','esvm',false,false);
esvm_train('cnn','esvm',false,true);
esvm_train('cnn','esvm',true,false);
esvm_train('cnn','esvm',true,true);

esvm_train('cnnhog','svm', false, true);
esvm_train('cnnhog','esvm',false,false);
esvm_train('cnnhog','esvm',false,true);
esvm_train('cnnhog','esvm',true,false);
esvm_train('cnnhog','esvm',true,true);