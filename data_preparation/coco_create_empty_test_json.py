import json
import os
from PIL import Image
from datetime import datetime

img_dir = 'Chula-ParasiteEgg-11/test/data/'
classes = ('Ascaris lumbricoides', 'Capillaria philippinensis', 'Enterobius vermicularis', 'Fasciolopsis buski',
           'Hookworm egg', 'Hymenolepis diminuta', 'Hymenolepis nana', 'Opisthorchis viverrine',
           'Paragonimus spp', 'Taenia spp. egg', 'Trichuris trichiura')
dataset_name = 'ParEgg'

#list all images in directory
img_paths = [ [f.path, f.name] for f in os.scandir(img_dir) if f.name.split('.')[-1] in ('jpg', 'png', 'jpeg', 'bmp')]
images = []
id = 1
for img_path, img_name in img_paths:
    img = Image.open(img_path)
    width, height = img.size
    images.append({'id': id,
                   'file_name': img_name,
                   'height': height,
                   'width': width,
                   'license': None,
                   'coco_url': None})
    id+=1
#add some general info
info = {'date': datetime.today().strftime('%Y-%m-%d'),
        'author': 'tureckova',
        'describtion': dataset_name}
categories = []
for ind, cls in enumerate(classes):
    categories.append({'id': ind,
                       'name': cls})
coco_json = {'info': info,
             'licenses':[],
             'categories': categories,
             'images': images,
             'annotations': []}
#save output json file
with open(img_dir+'test.json','w') as file:
    json.dump(coco_json, file)
