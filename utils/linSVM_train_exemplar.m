function m = linSVM_train_exemplar(m, features, labels, params)
% Actual training of an exemplar-SVM.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/
  
disp('********* Start train linear SVM for exemplar  *********');

weights = ones(size(features,1),1);
weights(1) = params.training_params.train_positives_constant;
c = params.training_params.train_svm_c;
m.svm_model = fitcsvm(features, labels,'PolynomialOrder', ...
                        [],'BoxConstraint', 2^c, 'KernelFunction', 'linear', 'KernelScale',1,...
                       'Standardize', 1,'ClassNames', [-1; 1], 'Weights', weights);

[~, maxpos] = predict(m.svm_model, m.x);
fprintf(1,' --- Max positive is %.3f\n',maxpos(1, 2));
end