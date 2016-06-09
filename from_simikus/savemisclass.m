function savemisclass(labels_test, labels_predicted, data, classes, prefix)

misclass = find(~(labels_test == labels_predicted));

m = 0;
root_imdir = strcat('misclassim_', prefix);
for i = (misclass')
    dest_dir = fullfile(root_imdir, char(classes(labels_test(i))));
    if ~exist(dest_dir,'dir')
        mkdir(dest_dir);
    end
    copyfile(data(i).imgfname, dest_dir);
    m = m+1;
end
fprintf('Total %i misclassified images files', m);
if misclass > 0
    fprintf(' copied to folder:\n%s\n', fullfile(pwd,root_imdir)); 
else
    fprintf('\n');
end