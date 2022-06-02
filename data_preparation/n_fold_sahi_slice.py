from sahi.slicing import slice_coco
import os

data_root = 'Chula-ParasiteEgg-11/data'
n_fold_folder = 'Chula-ParasiteEgg-11/5-fold-train-val/'
slice_height = 960
slice_width = 1280
overlap_height_ratio = 0.3
overlap_width_ratio = 0.3
min_area_ratio = 0.3
ignore_negative_samples = True

for dir in os.listdir(n_fold_folder):
    for split in ['train', 'val']:
        coco_dict, coco_path = slice_coco(
            coco_annotation_file_path=n_fold_folder+dir+'/'+split+'.json',
            image_dir=data_root,
            output_coco_annotation_file_name=n_fold_folder+dir+'/cut_'+str(slice_height)+'x'+str(slice_width)+'_'+split,
            output_dir=n_fold_folder+dir+'/cut_'+str(slice_height)+'x'+str(slice_width)+'_'+split,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            verbose=True,
            ignore_negative_samples=ignore_negative_samples,
            min_area_ratio=min_area_ratio
        )
    file_slicing_info = n_fold_folder+dir+'/cut_'+str(slice_height)+'x'+str(slice_width)+'.txt'
    with open(file_slicing_info, 'w') as file:
        file.write('data_root='+data_root+'\n')
        file.write('n_fold_folder='+n_fold_folder+'\n')
        file.write('slice_height='+str(slice_height)+'\n')
        file.write('slice_width='+str(slice_width)+'\n')
        file.write('overlap_height_ratio='+str(overlap_height_ratio)+'\n')
        file.write('overlap_width_ratio='+str(overlap_width_ratio)+'\n')
        file.write('min_area_ratio='+str(min_area_ratio)+'\n')
        file.write('ignore_negative_samples='+str(ignore_negative_samples)+'\n')
