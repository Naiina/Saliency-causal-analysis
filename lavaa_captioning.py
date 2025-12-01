print("start to run")
import os
import pandas as pd
import shutil
from transformers import  AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
from qwen_vl_utils import process_vision_info


def image_captioning_llava(image_folder,csv_save):

    l_caption = []
    l_img = []

    model_path = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

    # default: Load the model on the available device(s)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype="auto", device_map="auto", local_files_only=True, trust_remote_code=True
)


    # default processer
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True
)
    list_pict = os.listdir(image_folder)
    list_pict.sort()
    for img_file in tqdm(list_pict):
        img_path = os.path.join(image_folder, img_file)
        if ".png" in img_file:
            #promt = "Describe this picture in one sentence."
            #output_text = main_llava(img_path,processor,model,promt,100)
            #print(output_text)

            #promt = "Describe this picture in detail."
            #output_text = main_llava(img_path,processor,model,promt,100)
            #print(output_text)

            promt = "Describe this picture in one sentence. The sentence shoudn't start with: In this picture ... "
            output_text = main_llava(img_path,processor,model,promt)
            #print(output_text)


            l_caption.append(output_text)
            l_img.append(img_file)
            #print(output_text)

    d = {"img":l_img,"caption":l_caption}
    df = pd.DataFrame.from_dict(d)
    df.to_csv(csv_save)


def main_llava(img_path,processor,model,promt,max_tok = 100):
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                        },
                        {"type": "text", "text": promt},
                    ],
                }
            ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_tok)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text





   




def hoi_llava(image_folder,outfolder):

    l_caption = []
    l_img = []

    model_path = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

    # default: Load the model on the available device(s)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype="auto", device_map="auto", local_files_only=True, trust_remote_code=True
)


    # default processer
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True
)
    list_pict = os.listdir(image_folder)
    list_pict.sort()
    for img_file in tqdm(list_pict):
        img_path = os.path.join(image_folder, img_file)
        obj = img_file.split("_")[1]
        print(obj)
        if ".png" in img_file:
            if "no_hoi" in img_file:
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                        },
                        {"type": "text", "text": f"Is the person interacting with the {obj}? Answer with yes or no."},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=1)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            l_caption.append(output_text)
            l_img.append(img_file)
            if output_text[0].lower() == "yes":
                img_path_hoi_out2 = os.path.join(outfolder, img_file) 
                img_path_no_hoi_out2 = os.path.join(outfolder, img_file.split(".")[0]+"no_hoi.png") 
                shutil.copy(img_path, img_path_hoi_out2)
                shutil.copy(img_path.split(".")[0][:-3]+"no_hoi.png", img_path_no_hoi_out2)


    #d = {"img":l_img,"caption":l_caption}
    #df = pd.DataFrame.from_dict(d)
    #df.to_csv(csv_save)



def changed_llava(image_folder,outfolder):

    l_caption = []
    l_img = []

    model_path = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

    # default: Load the model on the available device(s)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype="auto", device_map="auto", local_files_only=True, trust_remote_code=True
)


    # default processer
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True
)
    list_pict = os.listdir(image_folder)
    list_pict.sort()
    for img_file in tqdm(list_pict):
        img_path = os.path.join(image_folder, img_file)
        obj = img_file.split("_")[1]
        print(obj)
        print(img_file)
        if ".png" in img_file:
            #if "person_small" in img_file:
            if ".png" in img_file:  
               
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img_path,
                            },
                            #{"type": "text", "text": f"Is there the human figure a toy or a real person? Answer with real or toy."},
                            {"type": "text", "text": f"Is the {obj} floating? Reply yes or no."},
                        ],
                    }
                ]

                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=1)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                print(output_text)
                l_caption.append(output_text)
                l_img.append(img_file)
                if output_text[0].lower() == "yes":
                    img_path_hoi_out2 = os.path.join(outfolder, img_file) 
                    #img_path_no_hoi_out2 = os.path.join(outfolder, img_file.split(".")[0]+"no_hoi.png") 
                    shutil.copy(img_path, img_path_hoi_out2)
                #    shutil.copy(img_path.split(".")[0][:-3]+"no_hoi.png", img_path_no_hoi_out2)






feat = "hoi"
dataset = "val2014"
if feat == "hoi":
    csv_save = f"COCO/{dataset}/hoi_detect_captions_standing_h.csv"
    image_folder = f"COCO/{dataset}/hoi_out_m_filter"
if feat == "size":
    csv_save = f"COCO/{dataset}/changed_temp_filtered_captions.csv"
    image_folder = f"COCO/{dataset}/final_changed"
image_captioning_llava(image_folder,csv_save)
#changed_llava(image_folder,"COCO/val2017/toy_person")
#hoi_llava("COCO/val2014/hoi_out_filtered","COCO/val2014/hoi_out_detect")