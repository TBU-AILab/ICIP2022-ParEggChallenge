import json
import os

test_json = 'Chula-ParasiteEgg-11/test/test.json'
main_folder = 'Chula-ParasiteEgg-11/models/TOOD_r101/'

with open(test_json, 'r') as file:
    test = json.load(file)

img_ids = [i for i in range(1, len(test['images'])+1)]

subfolders = [f.name for f in os.scandir(main_folder) if f.is_dir()]
results = []
for subfolder in subfolders:
    #add results from full size images
    json_file = main_folder + subfolder + '/test/results.noslicing.0.05.bbox.json'
    with open(json_file,'r') as file:
        result_noslicing=json.load(file)
    for ann in result_noslicing:
        ann['score'] = ann['score']+1
    results += result_noslicing
    #add results from sahi
    json_file = main_folder + subfolder + '/test-sliced/result.json'
    with open(json_file, 'r') as file:
        results += json.load(file)

with open(main_folder+'results.bbox.rowinput.json','w') as file:
    json.dump(results, file)
