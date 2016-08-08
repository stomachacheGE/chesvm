function filer = generate_random_test_img_filer(datasets_info)

num_cls = length(datasets_info);
cls_idx = randi(num_cls,1);
num_img_in_cls = length(datasets_info{cls_idx}.test_image_files);
img_idx = randi(num_img_in_cls,1);
filer = datasets_info{cls_idx}.test_image_files{img_idx};

end