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
    img_folder = "COCO/val2017/obj_to_change"
    out_folder = "COCO/val2017/changed_obj_6"
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



def qwen_inf_hoi():
    img_folder = "COCO/val2017/hoi"
    out_folder = "COCO/val2017/hoi_out"
    with open("img_and_promts_hoi_small.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in tqdm(reader):
                
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
                "num_inference_steps": 10,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():

                output = pipeline(**inputs)
                output_image_rm = output.images[0]
                output_image_rm.save(f"{out_folder}/{img_name.split(".")[0]}_{obj}_hoi.png")


        print("✅ Inference done.")




def qwen_inf_hoi_2():
    img_folder = "COCO/val2017/changed_obj_5"
    out_folder = "COCO/val2017/hoi_out_5"
    l_img = os.listdir(img_folder)


    for img_name in tqdm(l_img):
        print(img_name)
        if "rm" in img_name or "small" in img_name:
            continue
        obj = img_name.split(".")[0].split("_")[1]
        print(obj)
        
        img_path = os.path.join(img_folder,img_name)

        input_image = Image.open(img_path)
        if input_image.mode not in ("RGB", "RGBA"):
            input_image = input_image.convert("RGB")
        input_image.save(f"{out_folder}/{img_name.split(".")[0]}_no_hoi.png")

        #inputs = {
        #    "image": [output_image_rm],
        #    "prompt": f"Add a person using or interacting with a {obj}.",
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
        #    output_image_rm.save(f"{out_folder}/{img_name.split(".")[0]}_hoi.png")

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
            output_image_rm.save(f"{out_folder}/{img_name.split(".")[0]}_hoi.png")

        

    print("✅ Inference done.")







qwen_inf_hoi_2()

