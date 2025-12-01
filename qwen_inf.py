print("start to run")
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
#from diffusers import QwenImageEditPipeline
#from qwen_vl_utils import process_vision_info
import csv
from tqdm import tqdm
import pandas as pdQwenImageEditPipeline
import torchvision.transforms as T


def load_qwen_model():
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        device_map="balanced",        
        offload_folder="offload",
        local_files_only=True,

    )
    print("Pipeline loaded across GPUs")
    pipeline.set_progress_bar_config(disable=None)
    pipeline.enable_attention_slicing()
    

    return pipeline


print("\nRunning")
pipeline = load_qwen_model()
print("qwen loaded")



def qwen_inf_add_test():
    l_already_done = os.listdir("COCO/val2017/coco_modif_03_11_sorted/")+os.listdir("COCO/val2017/coco_modif2_sorted/")+os.listdir("COCO/val2017/coco_modif3")
    l_already_done = set(elem.split("_")[0] for elem in l_already_done)
    d_image_folder = {"outdoor":"COCO/val2017/coco_outdoor_small/","indoor":"COCO/val2017/coco_indoor_small/"}
    out_folder = "COCO/val2017/coco_modif_speedup_test"

    with open("img_and_promts.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in tqdm(reader):
                
            img_name = row[1]
            obj = row[2]
            scene = row[3]

            img_path = os.path.join(d_image_folder[scene],img_name)
            if img_name in l_already_done:
                continue

            input_image = Image.open(img_path)
            if input_image.mode not in ("RGB", "RGBA"):
                input_image = input_image.convert("RGB")
            device = pipeline.device or torch.device("cuda")

            input_image = input_image.to(device)#, dtype=torch.float16)
            

            inputs = {
                "image": [input_image],
                "prompt": f"Add a {obj}.",
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 10,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():

                output = pipeline(**inputs)
                output_image = output.images[0]
                output_image.save(f"{out_folder}/{img_name.split(".")[0]}_{obj}.png")


            inputs = {
                "image": [input_image],
                "prompt": [f"Add a small {obj} in the background."],#,"Add a dog"],
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 10,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }
            with torch.inference_mode():

                output = pipeline(**inputs)
                output_image = output.images[0]
                output_image.save(f"{out_folder}/{img_name.split(".")[0]}_small_{obj}.png")

        print("✅ Inference done.")

def qwen_inf_change():
    l_already_done = os.listdir("COCO/val2017/changed_obj_5/")
    l_already_done = set(elem.split("_")[0] for elem in l_already_done)
    l_already_done2 = os.listdir("COCO/val2017/changed_obj_6/")
    l_already_done2 = set(elem.split("_")[0] for elem in l_already_done2)
    img_folder = "COCO/val2017/obj_to_change"
    out_folder = "COCO/val2017/changed_obj_7"
    with open("img_and_promts_to_change_val2017.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for i,row in tqdm(enumerate(reader)):
            if i<289:
                continue
            img_name = row[1]
            obj = row[2]

            img_path = os.path.join(img_folder,img_name)
            if img_name in l_already_done:
                continue

            input_image = Image.open(img_path)
            if input_image.mode not in ("RGB", "RGBA"):
                input_image = input_image.convert("RGB")
            
            inputs = {
                "image": [input_image],
                "prompt": f"Remove the {obj}.",
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 10,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():

                output = pipeline(**inputs)
                output_image_rm = output.images[0]
                output_image_rm.save(f"{out_folder}/{img_name.split(".")[0]}_{obj}_rm.png")


            inputs = {
                "image": [output_image_rm],
                "prompt": [f"Add a {obj}. It has to be small or in the background."],
                #"prompt":[f"Add a {obj}. Make the {obj} integrated into the image as a non-central object, and have it's size scaled appropriately to its place in the foreground/background"],
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 15,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }
            with torch.inference_mode():
            
                output = pipeline(**inputs)
                output_image = output.images[0]
                output_image.save(f"{out_folder}/{img_name.split(".")[0]}_{obj}_small.png")

            inputs = {
                "image": [output_image_rm],
                "prompt": f"Add a {obj}.",
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 10,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():

                output = pipeline(**inputs)
                output_image = output.images[0]
                output_image.save(f"{out_folder}/{img_name.split(".")[0]}_{obj}.png")

        print("✅ Inference done.")



def qwen_inf_hoi(dataset):
    img_folder = f"COCO/{dataset}/hoi"
    out_folder = f"COCO/{dataset}/hoi_out_2"
    with open(f"COCO/{dataset}/img_and_promts_hoi.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in tqdm(reader):
            if int(row[0])<1005:
                continue
                
            img_name = row[1]
            obj = row[2]

            img_path = os.path.join(img_folder,img_name)

            input_image = Image.open(img_path)
            if input_image.mode not in ("RGB", "RGBA"):
                input_image = input_image.convert("RGB")
            input_image.save(f"{out_folder}/{img_name.split(".")[0]}_{obj}_no_hoi.png")
            
            inputs = {
                "image": [input_image],
                "prompt": f"Add a person interacting with the {obj}.",
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 15,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():

                output = pipeline(**inputs)
                output_image_rm = output.images[0]
                output_image_rm.save(f"{out_folder}/{img_name.split(".")[0]}_{obj}_hoi.png")
            
            #inputs = {
            #    "image": [input_image],
            #    "prompt": f"Add a person standing on the side.",
            #    "generator": torch.manual_seed(0),
            #    "true_cfg_scale": 4.0,
            #    "negative_prompt": " ",
            #    "num_inference_steps": 15,
            #    "guidance_scale": 1.0,
            #    "num_images_per_prompt": 1,
            #}

            #with torch.inference_mode():

            #    output = pipeline(**inputs)
            #    output_image_rm = output.images[0]
            #    output_image_rm.save(f"{out_folder}/{img_name.split(".")[0]}_{obj}_no_hoi_human.png")


        print("✅ Inference done.")






def qwen_inf_hoi_other(dataset):
    img_folder = f"COCO/{dataset}/hoi"
    out_folder = f"COCO/{dataset}/hoi_out_test"
    already_done = [elem.split("_")[0] for elem in os.listdir(out_folder)]
    with open(f"COCO/{dataset}/img_and_promts_hoi_other.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in tqdm(reader):
                
            img_name = row[1]
            if img_name.split("_")[0]+".png" in already_done:
                continue
            obj = row[3]

            img_path = os.path.join(img_folder,img_name)

            input_image = Image.open(img_path)
            if input_image.mode not in ("RGB", "RGBA"):
                input_image = input_image.convert("RGB")
            input_image.save(f"{out_folder}/{img_name.split(".")[0]}_{obj}_no_hoi.png")
            
            inputs = {
                "image": [input_image],
                "prompt": f"Add a person interacting with the {obj}.",
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 15,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():

                output = pipeline(**inputs)
                output_image_rm = output.images[0]
                output_image_rm.save(f"{out_folder}/{img_name.split(".")[0]}_{obj}_hoi_other.png")


        print("✅ Inference done.")



def qwen_inf_hoi_human(dataset):
    img_folder = f"COCO/{dataset}/hoi_out_m_filter"
    out_folder = f"COCO/{dataset}/hoi_out_h2"
    #already_done = [elem.split("_")[0] for elem in os.listdir(out_folder)]
    l_img_files = os.listdir(img_folder)
    for img_file in l_img_files:
        if "no_hoi" in img_file:
            img_path = os.path.join(img_folder,img_file)

            input_image = Image.open(img_path)
            if input_image.mode not in ("RGB", "RGBA"):
                input_image = input_image.convert("RGB")
            input_image.save(f"{out_folder}/{img_file}")
            
            inputs = {
                "image": [input_image],
                "prompt": f"Add a person standing on the side.",
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 15,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():

                output = pipeline(**inputs)
                output_image_rm = output.images[0]
                output_image_rm.save(f"{out_folder}/{img_file.split(".")[0]}_human.png")


    print("✅ Inference done.")


dataset = "val2014"
qwen_inf_hoi(dataset)

