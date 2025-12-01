print("start to run")
import os
import torch
from PIL import Image
from ultralytics import YOLO
import math
from tqdm import tqdm
#from accelerate import infer_auto_device_map
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


def correct_categ_plotting(img_path,matched_boxes,best_box,outpath):

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
    best_box = None         
    matched_boxes = []     

    
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

    l_files = list(set([ elem.split(".")[0].split("_")[0]+elem.split(".")[0].split("_")[1] for elem in os.listdir(image_folder) if elem.endswith(".png")]))
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


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2]+boxA[0], boxB[2]+boxB[0])
    yB = min(boxA[3]+boxA[1], boxB[3]+ boxB[1])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    

    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = boxA[2] * boxA[3] 
    boxBArea = boxB[2] * boxB[3]

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    #print("inter",interArea)
    #   print("iou",iou)

    # return the intersection over union value
    return iou




def nb_human_nb_obj(result,img_file):
    bbox = None
    area = None
    nb_person = 0
    nb_entity = 0

    for box in result.boxes:
        entity = result.names[int(box.cls)]  
        if entity == "person":
            nb_person+=1
        if entity in img_file:
            bbox = box.xywhn[0].tolist() 
            nb_entity+=1 
            area = bbox[2]*bbox[3]
            
    return nb_person,nb_entity,bbox,area


def yolo_segmentation_filter_hoi(image_folder, outfolder,outfolder2):
    model = YOLO("../huggingface_cache/yolov8s-world.pt")

    l_files = list(set([ elem.split(".")[0].split("_")[0]+"_"+elem.split(".")[0].split("_")[1]+"_"+elem.split(".")[0].split("_")[2]+"_"+elem.split(".")[0].split("_")[3] for elem in os.listdir(image_folder) if elem.endswith(".png")]))
    s_files = set(os.listdir(image_folder))
    for img_file in l_files: 
          
        img_file_hoi = img_file + "_hoi.png"
        img_file_no_hoi = img_file + "_no_hoi.png"
        img_path_hoi = os.path.join(image_folder, img_file_hoi) 
        img_path_no_hoi = os.path.join(image_folder, img_file_no_hoi)
        img_path_hoi_out = os.path.join(outfolder, img_file_hoi) 
        img_path_no_hoi_out = os.path.join(outfolder, img_file_no_hoi) 
        img_path_hoi_out2 = os.path.join(outfolder2, img_file_hoi) 
        img_path_no_hoi_out2 = os.path.join(outfolder2, img_file_no_hoi) 
        
        if img_file_hoi in s_files and img_file_no_hoi in s_files:

            results_h = model(img_path_hoi)
            results_n = model(img_path_no_hoi)

            nb_person_h,nb_entity_h,bbox_h,area_h = nb_human_nb_obj(results_h[0],img_file_hoi)
            nb_person_n,nb_entity_n,bbox_n,area_n = nb_human_nb_obj(results_n[0],img_file_no_hoi)
            #print(area_h,area_n)
            #print("person",nb_person_h,nb_person_n)
            #print("entity",nb_entity_h,nb_entity_n)
            if bbox_h!=None and bbox_n!=None:
                iou = bb_intersection_over_union(bbox_h, bbox_n)
                if nb_person_h == 1 and nb_person_n == 0 and nb_entity_h == 1 and nb_entity_n == 1:
                    if iou > 0.5:
                        shutil.copy(img_path_hoi, img_path_hoi_out)
                        shutil.copy(img_path_no_hoi, img_path_no_hoi_out)
                        #plotted_img = results_h[0].plot()  
                        #Image.fromarray(plotted_img[..., ::-1]).save(img_path_hoi_out)  
                        #plotted_img = results_n[0].plot()  
                        #Image.fromarray(plotted_img[..., ::-1]).save(img_path_no_hoi_out) 
                    else:
                        #plotted_img = results_h[0].plot()  
                        #Image.fromarray(plotted_img[..., ::-1]).save(img_path_hoi_out2)  
                        #plotted_img = results_n[0].plot()  
                        #Image.fromarray(plotted_img[..., ::-1]).save(img_path_no_hoi_out2) 
                        shutil.copy(img_path_hoi, img_path_hoi_out2)
                        shutil.copy(img_path_no_hoi, img_path_no_hoi_out2)
                    








feat = "hoi"
dataset = "val2014"
if feat == "hoi":    
    img_folder = f"COCO/{dataset}/hoi_out"
    outfile = f"COCO/{dataset}/hoi_out_filtered"
    outfile2 = f"COCO/{dataset}/hoi_out_dfiltered"

if feat == "size":
    img_folder = f"COCO/{dataset}/changed_obj_5"
    #outfile = "COCO/val2017/hoi_out_5"



#yolo_segmentation_filter_size(img_folder,outfile)
yolo_segmentation_filter_hoi(img_folder, outfile,outfile2)