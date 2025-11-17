# Saliency-causal-analysis

| Script                          | Description                                                | Output folder  / file                               |
|----------------------------------------|------------------------------------------------------------|-------------------------------------------------|
| `coco_image_selection_eddition.py`   | Identify appropriate pictures an object to change  | `img_and_promts_to_change_val2017.csv`        |
| `qwen_inf.py`                           | Generates image pairs by modifying selected objects.    | `COCO/val2017/changed_obj_i/`        |
| `yolo_segmentation.py` | Runs YOLO segmentation and keeps only objects with significant size ratio. |    `COCO/val2017/changed_i_yolo_filtered/`         |
| `laava_captioning.py`                  | Generates captions consistent with modified images.        | `COCO/val2017/changed_5_filtered_captions.csv`      |
| `feature_extraction_llm_caption.py` `save_feat`                 | extracts rank voice deprel     | `img_and_captions_changed_5.csv`      |
| `plot.py`                | rank plot and deprel Stuart-Maxwell  test    |       |


