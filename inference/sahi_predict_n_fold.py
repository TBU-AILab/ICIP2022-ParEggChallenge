import os
from sahi.predict import predict

model_path='models/FasterRCNN-r50-train-val/'
image_dir = 'Chula-ParasiteEgg-11/test/data/'
dataset_json = 'Chula-ParasiteEgg-11/test/test.json'
slice_width = 1280
slice_height = 960
eval_type = 'bbox'
name = 'test-sliced'

subfolders = [f.name for f in os.scandir(model_path) if f.is_dir()]
for subfolder in subfolders:
    if not os.path.exists(model_path+subfolder+'/'+name+'/result.json'):
        predict(
            slice_width=slice_width,
            slice_height=slice_height,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            model_confidence_threshold=0.25,
            source=image_dir,
            model_path=os.path.join(model_path,subfolder,'epoch_24.pth'),
            model_config_path=os.path.join(model_path,subfolder,'faster_rcnn_r50-parasiteEgg.py'),#change config name
            model_type='mmdet',
            postprocess_type='GREEDYNMM',
            postprocess_match_threshold=0.5,
            postprocess_match_metric='IOS',
            no_standard_prediction=False,
            export_visual=False,
            visual_bbox_thickness=1,
            visual_text_thickness=1,
            dataset_json_path=dataset_json,
            project=os.path.join(model_path+subfolder),
            name=name,
            return_dict=True)

# copy sript to output folder
import shutil
import os     # optional: for extracting basename / creating new filepath
import time   # optional: for appending time string to copied script
# generate filename with timestring
copied_script_name = time.strftime("%Y-%m-%d_%H%M") + '_' + os.path.basename(__file__)
shutil.copy(__file__, model_path+subfolder + os.sep + name + os.sep+ copied_script_name)
