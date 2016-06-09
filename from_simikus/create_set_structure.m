% CREATE_SET_STRUCTURE 
%
% Input:
% ------
% main_dir: 
% Ntrain:   
% Ntest:    
% file_ext: 
%
% Output:
% ------
% data:     
%
% -------------------------------------------------------------------------
% Author:   Steffen Wulf
% Date:     21.09.15
% Modified: 12.04.16 (Simkus)
% remarks:  < tested >
% -------------------------------------------------------------------------
function data = create_set_structure(main_dir,Ntrain,Ntest,file_ext,out_cnn_name)
    category_dirs = dir(main_dir);
 
    %remove '..' and '.' directories
    category_dirs(~cellfun(@isempty, regexp({category_dirs.name}, '\.*')))=[];
    category_dirs(strcmp({category_dirs.name},'split.mat'))=[]; 
    disp('*** Select train/test images from the data set ***');
    for c = 1:length(category_dirs)
        if isdir(fullfile(main_dir,category_dirs(c).name)) && ~strcmp(category_dirs(c).name,'.') ...
                && ~strcmp(category_dirs(c).name,'..')
            imgdir = dir(fullfile(main_dir,category_dirs(c).name, 'train', ['*.' file_ext]));          
            data(c).n_images_train = length(imgdir);
            data(c).classname = category_dirs(c).name;
            data(c).files_train = cellfun(@(string) ['train/' string], {imgdir(:).name}, 'UniformOutput', false); % prefix every cell entry with 'train/'
            data(c).feats_train = cellfun(@(string) ['train/feats_' out_cnn_name '/' string], {imgdir(:).name}, 'UniformOutput', false); % prefix every cell entry with 'train/'            
            fprintf(' -Image class %s:\n', data(c).classname);
            if Ntrain > 0 % choose random num training images                
                ids = randperm(length(imgdir));
                if Ntrain > numel(ids), Ntrain = numel(ids); end
                data(c).train_id = false(1,data(c).n_images_train);
                data(c).train_id(ids(1:Ntrain))=true;
                fprintf('   %i random selected training images out of %i\n', length(find(data(c).train_id)), length(data(c).files_train));
            else % take all training images in that class
                ids = 1:length(imgdir);
                data(c).train_id = true(1,data(c).n_images_train);
                fprintf('   %i training images\n', length(data(c).files_train));
            end
            
        end
        
        if isdir(fullfile(main_dir,category_dirs(c).name)) && ~strcmp(category_dirs(c).name,'.') ...
                && ~strcmp(category_dirs(c).name,'..')
            imgdir = dir(fullfile(main_dir,category_dirs(c).name, 'test', ['*.' file_ext]));           
            data(c).n_images_test = length(imgdir);
            data(c).classname = category_dirs(c).name;
            data(c).files_test = cellfun(@(string) ['test/' string], {imgdir(:).name}, 'UniformOutput', false); % prefix every cell entry with 'test/'
            data(c).feats_test = cellfun(@(string) ['test/feats_' out_cnn_name '/' string], {imgdir(:).name}, 'UniformOutput', false);
            if Ntest > 0  % choose random num test images
                ids = randperm(length(imgdir));
                if Ntest > numel(ids), Ntest = numel(ids); end
                data(c).test_id = false(1,data(c).n_images_test);
                data(c).test_id(ids(1:Ntest))=true;
                fprintf('   %i random selected test images out of %i\n', length(find(data(c).test_id)), length(data(c).files_test));
            else % take all test images in that class
                ids = 1:length(imgdir);
                data(c).test_id = true(1,data(c).n_images_test);
                fprintf('   %i test images\n', length(data(c).files_test));
            end
        end
        
        data(c).n_images = data(c).n_images_train + data(c).n_images_test;
        data(c).files = [data(c).files_train data(c).files_test];
        data(c).feats = [data(c).feats_train data(c).feats_test];
    end
end
