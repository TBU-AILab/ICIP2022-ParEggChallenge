import json
import time   # optional: for appending time string to copied script
import os


def results2submission(results_folder, results_json, output_suffix=''):

    test_json = 'Chula-ParasiteEgg-11/test/test.json'
    results_file = results_folder + results_json
    with open(test_json, 'r') as file:
        test = json.load(file)
    with open(results_file, 'r') as file:
        results = json.load(file)
    count=0
    ann_out={'annotations':[]}
    for ann in results:
        ann_out['annotations'].append({
            'id': count,
            'file_name': test['images'][ann['image_id']-1]['file_name'],
            'category_id': ann['category_id'],
            'bbox': ann['bbox']
        })
        count += 1

    # generate filename with timestring
    output_file_name = results_folder + 'A.I.Lab-TBU-CZ_' + time.strftime("%Y-%m-%d_%H%M") + output_suffix + '.json'
    with open(output_file_name, 'w') as file:
        json.dump(ann_out, file)


if __name__ == "__main__":
    results_folder = 'Chula-ParasiteEgg-11/models/FasterRCNN-r50/'
    results_json = 'results_mean_nms-one_class_img.json'
    results2submission(results_folder, results_json)

    # results_folder = 'Chula-ParasiteEgg-11/models/TOOD_r101-train-val/'
    # for results_json in [f.name for f in os.scandir(results_folder) if f.name[0:7] == 'results']:
    #     print(results_json)
    #     results2submission(results_folder, results_json, output_suffix=results_json.split('.')[0])
