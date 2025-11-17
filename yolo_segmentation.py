print("start to run")
import os
import torch
from PIL import Image
from ultralytics import YOLO
import math
from tqdm import tqdm
from accelerate import infer_auto_device_map
import pandas as pd
from collections import defaultdict
import shutil
import cv2
import numpy as np

torch.cuda.empty_cache()


def yolo_segmentation_plot(image_folder,segmented_images_dir):
    model = YOLO("../huggingface_cache/yolov8s-world.pt")

    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)

        results = model(img_path)

        for i, result in tqdm(enumerate(results)):
            plotted_img = result.plot()  
            output_path = os.path.join(segmented_images_dir, f"seg_{img_file}")
            Image.fromarray(plotted_img[..., ::-1]).save(output_path)  

        print(f"Processed and saved: {output_path}")


def correct_categ_plotting(img_path,matched_boxes,best_box,outpath,):

    img_file = img_path.split("/")[-1]
        
    orig_img = cv2.imread(img_path)
    img_drawn = orig_img.copy()

    for bbox_xyxy, conf in matched_boxes:
        x1, y1, x2, y2 = map(int, bbox_xyxy)

        if bbox_xyxy == best_box:
            color = (0, 255, 0)   # GREEN → selected detection
        else:
            color = (0, 0, 255)   # RED → rejected detection

        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            img_drawn,
            f"{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    # Save output
    output_path = os.path.join(outpath, f"seg_{img_file}")
    cv2.imwrite(output_path, img_drawn)




def find_obj_size(model, img_path, outpath,plot = False):
    results = model(img_path)
    img_file = img_path.split("/")[-1]

    best_area = None
    best_conf = -1
    best_box = None         # store best detection
    matched_boxes = []      # store all matching detections

    # FIRST PASS → Identify the best matching detection
    for result in results:
        for box in result.boxes:
            entity = result.names[int(box.cls)]
            conf = float(box.conf)

            if entity in img_path:
                bbox_xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                bbox_xywhn = box.xywhn[0].tolist()
                area = bbox_xywhn[2] * bbox_xywhn[3]

                matched_boxes.append((bbox_xyxy, conf))

                if conf > best_conf:
                    best_conf = conf
                    best_area = area
                    best_box = bbox_xyxy

    # If no match → return None
    if not matched_boxes:
        return 0

    if plot:
        correct_categ_plotting(img_path,matched_boxes,best_box,outpath)

    return best_area


def yolo_segmentation_filter_size(image_folder, outfolder,out_seg = None):
    model = YOLO("../huggingface_cache/yolov8s-world.pt")

    l_files = list(set([ elem.split(".")[0].split("_")[0]+"_"+elem.split(".")[0].split("_")[1] for elem in os.listdir(image_folder) if elem.endswith(".png")]))
    s_files = set(os.listdir(image_folder))
    for img_file in l_files:             

        img_path_small = os.path.join(image_folder, img_file) + "_small.png"
        img_path = os.path.join(image_folder, img_file) + ".png"
        img_path_small_out = os.path.join(outfolder, img_file) + "_small.png"
        img_path_out = os.path.join(outfolder, img_file) + ".png"

        if img_file + "_small.png" in s_files and img_file + ".png" in s_files:

            area_small = find_obj_size(model,img_path_small,out_seg)
            area_big = find_obj_size(model,img_path,out_seg)

            if area_small*area_big>0:
                if area_small *1.5 <area_big:
                    shutil.copy(img_path_small, img_path_small_out)
                    shutil.copy(img_path, img_path_out)


def yolo_segmentation_filter_hoi(image_folder, segmented_images_dir,csv_output):
    model = YOLO("../huggingface_cache/yolov8s-world.pt")

    l_img_id = [] 
    l_detected_obj = []
    l_bbox_size = []

    d = defaultdict(dict)
    

    os.makedirs(segmented_images_dir, exist_ok=True)

    for img_file in tqdm(os.listdir(image_folder)[:10], desc="Processing images"):
        img_path = os.path.join(image_folder, img_file)
        print("--------------------------------------------------")
        print(img_file)
        if os.path.isfile(img_path) and img_path.lower().endswith(".png"):
            print("enter loop")
            results = model(img_path)
            nb_entity = 0
            nb_person = 0 
            
            for result in tqdm(results):
                plotted_img = result.plot()  
                output_path = os.path.join(segmented_images_dir, f"seg_{img_file}")
                Image.fromarray(plotted_img[..., ::-1]).save(output_path) 
                

                for box in result.boxes:
                    entity = result.names[int(box.cls)]  
                    print(entity)
                    if entity == "person":
                        nb_person+=1
                    if entity in img_file:
                        nb_entity+=1
                        bbox = box.xywhn[0].tolist()  
                        area = bbox[2]*bbox[3]
                        mem_entity = entity
            if "no_hoi" in img_file:
                if nb_entity == 1 and nb_person == 0:    
                    l_img_id.append(img_file)
                    l_detected_obj.append(mem_entity)
                    l_bbox_size.append(area)
                    print("###########saved")
            else:
                if nb_entity == 1 and nb_person == 1:    
                    l_img_id.append(img_file)
                    l_detected_obj.append(entity)
                    l_bbox_size.append(area)
                    print("###########saved")
            
            print(nb_entity,nb_person)
    

    d = {"img":l_img_id,"detected obj":l_detected_obj,"size":l_bbox_size}
    df = pd.DataFrame(d)
    df.to_csv(csv_output, index=False)
    print(f"\n✅ Saved bounding box data to {csv_output}")


#image_folder = "COCO/val2017/coco_modif_03_11_sorted"
outfile = "COCO/val2017/changed_5_filtered"
img_folder = "COCO/val2017/changed_obj_5"

yolo_segmentation_filter_size(img_folder,outfile)