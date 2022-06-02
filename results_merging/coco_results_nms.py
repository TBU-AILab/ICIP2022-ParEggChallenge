import json
import pandas as pd
from nonMaximumSuppression import nms, mean_nms, np_encoder, get_img_ids, separate_image_results, pd_dict_to_results

out_folder = 'Chula-ParasiteEgg-11/models/FasterRCNN_r50/'
res_file = out_folder + 'results.bbox.rowinput.json'

with open(res_file, 'r') as file:
    results = json.load(file)

# nms applied only on same category detections
# use pandas to group by images and detected categories
pd_results = pd.DataFrame(results)
pd_results_img_sort = pd_results.groupby('image_id')
best_pred = []
best_one_pred = []
best_one_pred2 = []
for group_name, group in pd_results_img_sort:
    print('Processing img_id: ', group_name)
    group_category_sort = group.groupby('category_id')
    best_pred_img = []
    for group_category_name, group_category in group_category_sort:
        print('Processing category_id: ', group_category_name)
        # format to original list of dicts
        group_category_formated = pd_dict_to_results(group_category)
        if len(group_category_formated)>1:
            best_pred_img += mean_nms(group_category_formated, match_metric='IOU', match_threshold=0.5)
        else:
            best_pred_img += group_category_formated
    best_pred += best_pred_img
    if len(best_pred_img) > 1:
        score = []
        #category = []
        for pred in best_pred_img:
            score.append(pred['score'])
            #category.append(pred['category_name'])
        best_one_pred.append(best_pred_img[score.index(max(score))])
        best_category_id = best_pred_img[score.index(max(score))]['category_id']
        if best_category_id == 3: #in case of Fasciolopsis buski use only the best since the sides of ocullar gives false positives
            best_one_pred2.append(best_pred_img[score.index(max(score))])
        else:
            for pred in best_pred_img:
                if pred['category_id'] == best_category_id and pred['score'] > max(score)*0.75:
                    best_one_pred2.append(pred)
    else:
        best_one_pred += best_pred_img
        best_one_pred2 += best_pred_img

with open(out_folder+'results_mean_nms-one_class_img.json', 'w') as file:
    json.dump(best_one_pred2, file, default=np_encoder)

