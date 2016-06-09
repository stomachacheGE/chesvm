function esvm_initialize_results_folder(datasets_info)

res_dir = fullfile('.', params.results_folder);

res_folder = {'models', 'features'}

if ~exist(res_dir, 'dir')
    mkdir(res_dir);
    mkdir([res_dir '/models'];
    mkdir([res_dir '/features'];
    
    cls_names = {datasets_info(:).cls_name};
    
    for i=1:length(cls_names)
        filer_1 = fullpath(res_dir, 'models', cls_names{i});
        filer_2 = fullpath(res_dir, 'features', cls_names{i});
        mkdir(filer_1);
        mkdir(filer_2);
    end
    

fprintf(1,'Results floder already exists.');
end