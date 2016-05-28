function extract_features(dirname,file_ext,varargin)
%fu.    extract_features(fullfile('..',imgFolder,dataset_dir), desc_name, file_ext, '/train', cnn_path, out_cnn_name);
% for image files
if nargin > 3
    img_ext = cell2mat(varargin(1)); %fu. file_ext:jpg
    addFolder = cell2mat(varargin(2)); %fu. :/train
    cnn_path = cell2mat(varargin(3)); %fu. :cnn_path(model_path)
    last_cnn_name = cell2mat(varargin(4)); %fu. out_cnn_name:relu6
end

d = dir(dirname);
d = d(3:end); %remove . and .. dirs
d = d(cat(1,d.isdir)); % remove non dirs

for i=1:length(d)
    folder = fullfile(dirname,d(i).name,addFolder);    
    
    if strcmp(file_ext,'cnn')  
         % CNN FEATURES        
        detect_features_cnn(folder,file_ext,img_ext,last_cnn_name,cnn_path);
    % elseif OTHER METHOD       
    end
end

end