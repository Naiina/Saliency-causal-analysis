
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import matplotlib.image as mpimg
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sys import argv
import cv2
import csv
#from utils import get_dist_between_two_objects,rank_deprel_dep_nb, dep_dist
#from utils import get_dict_categ_to_supercateg, get_annot_categ, get_normalised_size_and_pos_categ, mean_masked_saliency
import json
from collections import Counter
import shutil
import random
from PIL import Image





# --- Output folders ---
os.makedirs('COCO/val2017/coco_outdoor', exist_ok=True)
os.makedirs('COCO/val2017/coco_indoor', exist_ok=True)

# --- Category groups ---
OUTDOOR_CATS = {
    'vehicle': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
    'outdoor': ['bench'],
}
INDOOR_CATS = {

    'furniture': ['chair', 'couch', 'potted plant', 'bed', 'dining table'],
    'appliance': ['microwave', 'oven', 'toaster', 'refrigerator'],
    'indoor': ['book', 'clock', 'vase'],

}



ITEM_TO_ADD_ALL = {"person":["person"],'animal': ['cat', 'dog'],'electronic': ['cell phone'],}
ITEM_TO_ADD_ALL_INDOOR = {'kitchen': ['bottle', 'wine glass', 'cup', 'bowl'],'furniture': ['chair', 'potted plant'],
'indoor': [ 'vase',]}


ITEM_TO_ADD_ALL_OUTDOOR = {"vehicle":['bicycle', 'car', 'motorcycle', 'bus', 'truck'], 
'outdoor': ['bench']}


# Flatten to sets for easy lookup
outdoor_labels = set(sum(OUTDOOR_CATS.values(), []))
indoor_labels = set(sum(INDOOR_CATS.values(), []))



def compute_coco_category_distribution(coco):
    """
    Compute normalized category frequencies (probabilities) from a COCO dataset.
    Returns:
        dict: {category_name: probability}
    """
    cat_counts = Counter()
    
    # Get all annotation IDs
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    
    # Count category occurrences
    for ann in anns:
        cat_id = ann['category_id']
        cat_name = coco.loadCats([cat_id])[0]['name']
        cat_counts[cat_name] += 1

    # Normalize to probabilities
    total = sum(cat_counts.values())
    cat_probs = {cat: count / total for cat, count in cat_counts.items()}
    
    return cat_probs



def random_choice(candidates, categ_stats):
    """
    Choose a random category from 'candidates' based on probabilities in 'categ_stats'.
    """
    # Extract probabilities for only the candidate categories
    probs = np.array([categ_stats.get(cat, 0.0) for cat in candidates], dtype=float)
    if probs.sum() == 0:
        probs = np.ones(len(candidates)) / len(candidates)
    else:
        probs /= probs.sum()  # normalize to sum=1

    return random.choices(list(candidates), weights=probs, k=1)[0]



def find_item_to_add(s_categ,ann,d_categ_freq):
    #l_obj = []
    
    unique = set(s_categ)-ann
    if unique:
        obj = random_choice(unique,d_categ_freq)
    #l_obj.append(obj)
        return obj
    return None


def find_item_to_change(l_ann,l_bbox):
    paired = list(zip(l_ann, l_bbox))
    random.shuffle(paired)
    d =  defaultdict(list)
    for ann,bbox in paired:
        d[ann].append(bbox)  

    #print(d)  
    for k,v in d.items():
        if len(v) == 1:
            if 0.01<v[0]<0.25:
                return k 
    return None


def find_another_item_to_change(l_ann,l_bbox,obj):
    paired = list(zip(l_ann, l_bbox))
    random.shuffle(paired)
    d =  defaultdict(list)
    for ann,bbox in paired:
        if ann != obj:
            d[ann].append(bbox)  

    #print(d)  
    for k,v in d.items():
        #if len(v) == 1:
        if 0.01<v[0]<0.05:
            return k 
    return None


def classify_and_save_add(s_categ_to_add_outdoor,s_categ_to_add_indoor,s_categ_if_table,s_to_exclude,max_it = -1):

    l_obj_to_add = []
    l_img = []
    l_scene = []
    
    for i,img_id in enumerate(coco.getImgIds()):
        if max_it>0 and i>max_it:
            break
        
        img = coco.loadImgs([img_id])[0]
        img_area = img['width'] * img['height']

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        num_objects = len(anns)
        cat_ids = [ann['category_id'] for ann in anns]
        cat_names = [coco.loadCats([cid])[0]['name'] for cid in cat_ids]

        

        if not set(cat_names)&s_to_exclude:

            # --- (1) More than 5 objects ---
            if num_objects < 5:
                pass

            # --- (2) No single object dominates ---
            cat_area = {}
            for ann in anns:
                x, y, w, h = ann['bbox']
                area_ratio = (w * h) / img_area
                cat_id = ann['category_id']
                cat_area[cat_id] = cat_area.get(cat_id, 0) + area_ratio

            # Find the largest category coverage
            max_cat_area_ratio = max(cat_area.values()) if cat_area else 0
            dominant_category = max_cat_area_ratio > 0.5
            if dominant_category:
                pass
            
            # --- (3) Scene classification ---
            
            
            
            indoor_count = sum(name in indoor_labels for name in cat_names)
            outdoor_count = sum(name in outdoor_labels for name in cat_names)
            
            if outdoor_count > indoor_count and outdoor_count>= 3:
                target_dir = "COCO/val2017/coco_outdoor"
                s_categ_to_add = set(s_categ_to_add_outdoor)
                if "table" in cat_names:
                    s_categ_to_add  |= s_categ_if_table

                item_to_add = find_item_to_add(s_categ_to_add,set(cat_names),d_categ_freq)
                dest_path = os.path.join(target_dir, img['file_name'])

                img_path = os.path.join(imgDir, img['file_name'])

                if os.path.exists(img_path):
                    shutil.copy(img_path, dest_path)
                
                    l_img.append(img['file_name'])
                    l_obj_to_add.append(item_to_add)
                    l_scene.append("outdoor")
            elif indoor_count > outdoor_count and indoor_count >= 3:
                target_dir = "COCO/val2017/coco_indoor"
                s_categ_to_add = set(s_categ_to_add_indoor)
                if "table" in cat_names:
                    s_categ_to_add  |= s_categ_if_table


                item_to_add = find_item_to_add(s_categ_to_add,set(cat_names),d_categ_freq)
                dest_path = os.path.join(target_dir, img['file_name'])

                img_path = os.path.join(imgDir, img['file_name'])

                if os.path.exists(img_path):
                    shutil.copy(img_path, dest_path)
                
                    l_img.append(img['file_name'])
                    l_obj_to_add.append(item_to_add)
                    l_scene.append("indoor")

        #img_path = os.path.join(imgDir, img['file_name'])

        #I = cv2.imread(img_path)
        #I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        #for ann in anns:
        #    x, y, w, h = ann['bbox']
        #    cv2.rectangle(I, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        #plt.imshow(I)
        #plt.axis('off')
        #plt.show()


    d = {"img":l_img,"obj_to_add":l_obj_to_add,"scene":l_scene}
    df = pd.DataFrame.from_dict(d)
    df.to_csv("img_and_promts_val2017.csv")








def classify_and_save_change(max_it = -1):

    l_obj_to_add = []
    l_img = []

    for i,img_id in enumerate(coco.getImgIds()):
        if max_it>0 and i>max_it:
            break
        
        img = coco.loadImgs([img_id])[0]
        img_area = img['width'] * img['height']

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        num_objects = len(anns)
        cat_ids = [ann['category_id'] for ann in anns]
        cat_names = [coco.loadCats([cid])[0]['name'] for cid in cat_ids]
        cat_bbox = [ann['bbox'][2]*ann['bbox'][2]/img_area for ann in anns]

        # --- (1) More than 5 objects ---
        if num_objects < 5:
            #print("too few obj")
            continue

        # --- (2) No single object dominates ---
        cat_area = {}
        for ann in anns:
            x, y, w, h = ann['bbox']
            area_ratio = (w * h) / img_area
            cat_id = ann['category_id']
            cat_area[cat_id] = cat_area.get(cat_id, 0) + area_ratio

        max_cat_area_ratio = max(cat_area.values()) if cat_area else 0
        dominant_category = max_cat_area_ratio > 0.5
        if dominant_category:
            continue
    
        target_dir = "COCO/val2017/obj_to_change/"

        item_to_add = find_item_to_change(cat_names,cat_bbox)
        #print(item_to_add)
        
        if item_to_add:
            #print("saved loop")
            l_img.append(img['file_name'])
            l_obj_to_add.append(item_to_add)
            img_path = os.path.join(imgDir, img['file_name'])

            if os.path.exists(img_path):
                img_ = Image.open(img_path)
                img_.thumbnail((500, 500))
                output_path = os.path.join(target_dir,img['file_name'])
                
                img_.save(output_path, optimize=True, quality=85)
                #shutil.copy(img_path, target_dir)

            #I = cv2.imread(img_path)
            #I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            #for ann in anns:
            #    x, y, w, h = ann['bbox']
            #    cv2.rectangle(I, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            #plt.imshow(I)
            #plt.axis('off')
            #plt.show()


    d = {"img":l_img,"obj_to_add":l_obj_to_add}
    df = pd.DataFrame.from_dict(d)
    df.to_csv("img_and_promts_to_change_val2017.csv")





def classify_and_save_change_2(max_it = -1):

    l_obj_to_add = []
    l_img = []

    for i,img_id in enumerate(coco.getImgIds()):
        if max_it>0 and i>max_it:
            break
        
        img = coco.loadImgs([img_id])[0]
        img_area = img['width'] * img['height']

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        num_objects = len(anns)
        cat_ids = [ann['category_id'] for ann in anns]
        cat_names = [coco.loadCats([cid])[0]['name'] for cid in cat_ids]
        cat_bbox = [ann['bbox'][2]*ann['bbox'][2]/img_area for ann in anns]

        # --- (1) More than 5 objects ---
        if num_objects < 5:
            #print("too few obj")
            continue

        # --- (2) No single object dominates ---
        cat_area = {}
        for ann in anns:
            x, y, w, h = ann['bbox']
            area_ratio = (w * h) / img_area
            cat_id = ann['category_id']
            cat_area[cat_id] = cat_area.get(cat_id, 0) + area_ratio

        max_cat_area_ratio = max(cat_area.values()) if cat_area else 0
        dominant_category = max_cat_area_ratio > 0.5
        if dominant_category:
            continue
    
        target_dir = "COCO/val2017/obj_to_change_2/"

        item_to_add = find_item_to_change_2(cat_names,cat_bbox)
        #print(item_to_add)
        
        if item_to_add:
            #print("saved loop")
            l_img.append(img['file_name'])
            l_obj_to_add.append(item_to_add)
            img_path = os.path.join(imgDir, img['file_name'])

            if os.path.exists(img_path):
                img_ = Image.open(img_path)
                img_.thumbnail((500, 500))
                output_path = os.path.join(target_dir,img['file_name'])
                
                img_.save(output_path, optimize=True, quality=85)
                #shutil.copy(img_path, target_dir)

            #I = cv2.imread(img_path)
            #I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            #for ann in anns:
            #    x, y, w, h = ann['bbox']
            #    cv2.rectangle(I, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            #plt.imshow(I)
            #plt.axis('off')
            #plt.show()


    d = {"img":l_img,"obj_to_add":l_obj_to_add}
    df = pd.DataFrame.from_dict(d)
    df.to_csv("img_and_promts_to_change_2_val2017.csv")


def get_anns_by_filename(coco, filename):
    # Look up COCO image entry by filename
  
    img_info = next(img for img in coco.dataset['images'] if img['file_name'] == filename)
    img_id = img_info['id']
    img_area = img_info['width'] * img_info['height']

    # Load annotations
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    return coco.loadAnns(ann_ids), img_area




def img_without_human(imgDir,dataset,max_it = -1):
    l_img = []
    l_obj = []
    for i,img_id in tqdm(enumerate(coco.getImgIds())):
        if max_it>0 and i>max_it:
            break
        
        img = coco.loadImgs([img_id])[0]
        img_area = img['width'] * img['height']

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        num_objects = len(anns)
        cat_ids = [ann['category_id'] for ann in anns]
        cat_names = [coco.loadCats([cid])[0]['name'] for cid in cat_ids]
        cat_bbox = [ann['bbox'][2]*ann['bbox'][2]/img_area for ann in anns]
        #print(i)
        if "person" in cat_names:
            #print("contains a person")
            continue

        # --- (1) More than 5 objects ---
        if num_objects < 5:
            #print("too few obj")
            continue

        # --- (2) No single object dominates ---
        cat_area = {}
        for ann in anns:
            x, y, w, h = ann['bbox']
            area_ratio = (w * h) / img_area
            cat_id = ann['category_id']
            cat_area[cat_id] = cat_area.get(cat_id, 0) + area_ratio

        # Find the largest category coverage
        max_cat_area_ratio = max(cat_area.values()) if cat_area else 0
        dominant_category = max_cat_area_ratio > 0.5
        if dominant_category:
            #print("print dominant categ")
            continue
    
        target_dir = f"COCO/{dataset}/hoi/"

        item_to_add = find_item_to_change(cat_names,cat_bbox)
        if item_to_add:

            #print(item_to_add)
            dest_path = os.path.join(target_dir, img['file_name'])

            img_path = os.path.join(imgDir, img['file_name'])
            #print(img_path)
            if os.path.exists(img_path):
                shutil.copy(img_path, dest_path)
        
        # 
            l_img.append(img['file_name'])
            l_obj.append(item_to_add)
    
            img_path = os.path.join(imgDir, img['file_name'])

            #I = cv2.imread(img_path)
            #I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            #for ann in anns:
            #    x, y, w, h = ann['bbox']
            #    cv2.rectangle(I, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            #plt.imshow(I)
            #plt.axis('off')
            #plt.show()
    d = {"img":l_img,"obj_to_add":l_obj}
    df = pd.DataFrame.from_dict(d)
    df.to_csv(f"COCO/{dataset}/img_and_promts_hoi.csv")






def hoi_another_obj(csv_file,dataset,coco,max_it = -1):
    l_img = []
    l_obj = []
    l_other_obj = []
    
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            filename = row[1]
            
            anns,img_area = get_anns_by_filename(coco, filename)

        
            cat_ids = [ann['category_id'] for ann in anns]
            cat_names = [coco.loadCats([cid])[0]['name'] for cid in cat_ids]
            cat_bbox = [ann['bbox'][2]*ann['bbox'][2]/img_area for ann in anns]

            new_item_to_add = find_another_item_to_change(cat_names,cat_bbox,row[2])
            if new_item_to_add:

                l_img.append(row[1])
                l_obj.append(row[2])
                l_other_obj.append(new_item_to_add)
        


            #I = cv2.imread(img_path)
            #I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            #for ann in anns:
            #    x, y, w, h = ann['bbox']
            #    cv2.rectangle(I, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            #plt.imshow(I)
            #plt.axis('off')
            #plt.show()
    d = {"img":l_img,"obj_to_add":l_obj,"other_obj":l_other_obj}
    df = pd.DataFrame.from_dict(d)
    df.to_csv(f"COCO/{dataset}/img_and_promts_hoi_other.csv")











s_categ_to_add_indoor = set(["person", 'cat', 'dog', 'potted plant'])
s_categ_if_table = set(["cup","bowl","book","bottle","vase",'cell phone'])


s_categ_to_add_outdoor = set(["person", 'cat', 'dog',
    'bicycle', 'car', 'motorcycle', 'bus', 'truck','bench'])


s_categ_to_exclude = set(["airplane","boat",'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',])




#img_without_human(-1)
#print("end")

#img_without_human("COCO/val2014","val2014",-1)
#classify_and_save_change_2(-1)


dataDir = 'COCO'
dataType = 'val2014'
imgDir = os.path.join(dataDir, dataType)

annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

d_categ_freq = compute_coco_category_distribution(coco)


dataset = "val2014"
csv_file = f"COCO/{dataset}/img_and_promts_hoi.csv"
hoi_another_obj(csv_file,dataset,coco,max_it = -1)