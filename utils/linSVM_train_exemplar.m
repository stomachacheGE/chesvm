function m = linSVM_train_exemplar(m, features, labels, params)

    %fprintf('Use %s feature with dimension: [%d %d] \n', feat_name, feat_size(1), feat_size(2));
    % Cross-validation with 5-fold (note the obtion -v 5)
    %C_vals=log2space(7,10,5);
    %C_vals = log2space(1,7,5);

    %{
    %# grid of parameters
    folds = 5;
    [C,gamma] = meshgrid(-5:2:15, -15:2:3);

    %# grid search, and cross-validation
    cv_acc = zeros(numel(C),1);
    for i=1:numel(C)
        cv_acc(i) = svmtrain(labels, features, ...
                        sprintf('-c %f -g %f -v %d', 2^C(i), 2^gamma(i), folds));
        fprintf('Parameter selection iteraiton %d with regularization C=%f, g=%f , accuracy=%f\n', i, 2^C(i), 2^gamma(i),cv_acc(i));
    end

    %# pair (C,gamma) with best accuracy
    [~,idx] = max(cv_acc);
    %{
    clear iter;
    for iter=1:length(C_vals);
        opt_string=['-t 0 -v 5 -c ' num2str(C_vals(iter))];
        xval_acc(iter)=svmtrain(labels, features,opt_string);
        fprintf('Parameter selection iteraiton %d with regularization C=%d , accuracy=%f\n', iter, C_vals(iter),xval_acc(iter));
    end

    % select the best C among
    [~,ind]=max(xval_acc);
    %}
    %}
    % Train the model with the feature vectors
    %fprintf('Choose regularization C=%d, strat training... \n',C_vals(ind));
    %linSVMmodel = svmtrain(labels,features,['-b 1 -t 0 -c ' num2str(C_vals(ind))]);
    %wpos = params.training_params.train_positives_constant;
    wpos = 100;
    
    disp('********* Start train linear SVM for exemplar  *********');
    %feat_size = size(features_cell{1});
    %fprintf('Use %s feature with dimension: [%d %d] \n', feat_name, feat_size(1), feat_size(2));
    % Cross-validation with 5-fold (note the obtion -v 5)
    %C_vals=log2space(7,10,5);
    %{
    C_vals = log2space(-5,5,2);
    clear i;
    for i=1:length(C_vals);
        %opt_string=['-t 0  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels, features, sprintf(['-s 0 -t 0 -v 5 -c %f -w1 %.9f -q'],...
                                                    num2str(C_vals(i)), wpos));
        fprintf('Parameter selection iteraiton %d with regularization C=%d , accuracy=%f\n', i, C_vals(i),xval_acc(i));
    end

    % select the best C among
    [~,ind]=max(xval_acc);
    
    %svm_model = svmtrain(labels,features,sprintf(['-s 0 -t 0 -c %f -w1 %.9f -q'],...
    %                                                params.training_params.train_svm_c, wpos));
    %fprintf(1,sprintf(['-s 0 -t 0 -c %f -w2 %.9f -q'],...
    %                                                params.training_params.train_svm_c, wpos));
    %fprintf('Train linear SVM model succeeds. \n');
    %}
        % Train the model with the feature vectors
%    fprintf('Choose regularization C=%d, strat training... \n',C_vals(ind));

    %svm_model = svmtrain(labels,features, sprintf(('-s 0 -t 0 -c 0.0003 -w1 %.9f -q'),...
    %                                                 wpos));
     weights = ones(size(features,1),1);
     weights(1) = 60;
     m.svm_model = fitcsvm(features, labels,'PolynomialOrder', ...
                                [],'BoxConstraint', 2^(-3), 'KernelFunction', 'linear', 'KernelScale',1,...
                               'Standardize', 1,'ClassNames', [-1; 1], 'Weights', weights);
     %m.svm_model = fitcsvm(features, labels,'PolynomialOrder', ...
     %                           [],'BoxConstraint', 2^(-3), 'KernelFunction', 'rbf', 'KernelScale','auto',...
     %                          'Standardize', 1,'ClassNames', [-1; 1], 'Weights', weights);
                
    %{
    if length(svm_model.sv_coef) == 0
      %learning had no negatives
      wex = m.w;
      b = m.b;
      fprintf(1,'reverting to old model...\n');
    else

      %convert support vectors to decision boundary
      svm_weights = full(sum(svm_model.SVs .* ...
                             repmat(svm_model.sv_coef,1, ...
                                    size(svm_model.SVs,2)),1));

      wex = svm_weights;
      b = svm_model.rho;

      %{
      %% project back to original space
      b = b + wex'*A(m.mask,:)'*mu(m.mask);
      wex = A(m.mask,:)*wex;

      wex2 = zeros(size(superx,1),1);
      wex2(m.mask) = wex;

      wex = wex2;
      %}
      %% issue a warning if the norm is very small
      if norm(wex) < .00001
        fprintf(1,'learning broke down!\n');
      end  
    end
    %}
    %fprintf(1,'dimension of wex is [%d %d]\n', size(wex,1),size(wex,2));
    %fprintf(1,'dimension of x is [%d %d]\n', size(m.x,1),size(m.x,2));
    %maxpos = wex*m.x' - b;
    [~, maxpos] = predict(m.svm_model, m.x);
    fprintf(1,' --- Max positive is %.3f\n',maxpos(1, 2));
    %fprintf(1,'SVM iteration took %.3f sec, ',toc(starttime));
%{
    m.w = reshape(wex, size(m.w));
    m.b = b;
    
      %convert support vectors to decision boundary
  svm_weights = full(sum(svm_model.SVs .* ...
                         repmat(svm_model.sv_coef,1, ...
                                size(svm_model.SVs,2)),1));
  

r = m.w*features(2:end,:)' - m.b;
%}
[~, r] = predict(m.svm_model, features(2:end,:));
r = r(:,2);
svs = find(r >= 0.0000);

if length(svs) == 0
  fprintf(1,' ERROR: number of negative support vectors is 0!\n');
  %error('Something went wrong');
end

fprintf(1,' %d of negative training data are falsly classified with this model\n',length(svs));

end