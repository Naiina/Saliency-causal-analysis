print("start to run")
import os
import torch
from PIL import Image
import pandas as pd

import os
from torch.cuda.amp import autocast

import math
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

import csv
from tqdm import tqdm
#from diffusers import FluxKontextPipeline
from accelerate import infer_auto_device_map

from qwen_vl_utils import process_vision_info


def image_captioning_llava(image_folder):

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

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": "Describe this image in detail in one sentence."},
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
        generated_ids = model.generate(**inputs, max_new_tokens=200)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        l_caption.append(output_text)
        l_img.append(img_file)
        print(output_text)

    d = {"img":l_img,"caption":l_caption}
    df = pd.DataFrame.from_dict(d)
    df.to_csv("COCO/val2017/changed_5_filtered_captions.csv")


image_folder = "COCO/val2017/changed_5_filtered"
image_captioning_llava(image_folder)