import json
from sklearn.model_selection import KFold, train_test_split
import os

def get_annotation_split(annotations, image_index):
    """annotations: coco dictionary with annotations
       image_index: index of images desired in output split"""
    img_list = []
    ann_list = []
    real_image_indexes = []
    for ind in image_index:
        real_image_indexes.append(annotations['images'][ind]['id'])
        img_list.append(annotations['images'][ind])
    for ann in annotations['annotations']:
        if ann['image_id'] in real_image_indexes:
            ann_list.append(ann)
    annotations_split = {'info': annotations['info'],
                         'licenses': annotations['licenses'],
                         'images': img_list,
                         'annotations': ann_list,
                         'categories': annotations['categories']
                         }
    return annotations_split

# file with all annotation
annotations_file = 'Chula-ParasiteEgg-11/labels.json'
annotations_file_cut = 'Chula-ParasiteEgg-11/cut_960x1280__coco.json'
output_folder = 'Chula-ParasiteEgg-11/'
n_splits = 5

# load annotations
with open(annotations_file,'r') as file:
    annotations = json.load(file)

base_dir = output_folder+str(n_splits)+'-fold/'
if not os.path.exists(base_dir): os.mkdir(base_dir)
images_index = [i for i in range(len(annotations['images']))]
kf = KFold(n_splits=n_splits, shuffle=True)
kf.get_n_splits(images_index)
counter = 0
for train_index, val_index in kf.split(images_index):
    train_annotations = get_annotation_split(annotations, train_index)
    val_annotations = get_annotation_split(annotations, val_index)
    split_dir = base_dir + 'fold-' + str(counter) + '/'
    if not os.path.exists(split_dir): os.mkdir(split_dir)
    with open(split_dir+'val.json', 'w') as val_file:
        json.dump(val_annotations, val_file)
    with open(split_dir+'train.json', 'w') as train_file:
        json.dump(train_annotations, train_file)
    counter +=1
