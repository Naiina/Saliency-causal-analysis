# Saliency-causal-analysis

| Script                          | Description                                                | Output folder  / file                               | run time |
|----------------------------------------|------------------------------------------------------------|-------------------------------------------------|--------|
| `coco_image_selection_eddition.py`   | Identify appropriate pictures an object to change  | `img_and_promts_to_change_val2017.csv`        |
| `qwen_inf.py`                           | Generates image pairs by modifying selected objects.    | `COCO/val2017/changed_obj_i/`        | 3min / image
| `yolo_segmentation.py` | For Rank only: Runs YOLO segmentation and keeps only objects with significant size ratio. |    `COCO/val2017/changed_i_yolo_filtered/`         | quick|
| `laava_captioning.py`                  | Generates captions consistent with modified images.        | `COCO/val2017/changed_5_filtered_captions.csv`      | 3s per item|
| `feature_extraction_llm_caption.py` `save_feat`                 | Extracts rank voice deprel     | `img_and_captions_changed_5.csv`      | |
| `plot.py`                | Rank plot and deprel Stuart-Maxwell  test    |       |

## Size modification subset 5:
- `changed_obj_5`: 289 pairs
-  `changed_5_yolo_filtered`: 200 pairs
-  Both ranks -1: 30   /   Small only mentioned: 3   /   Big only mentioned: 41   /   Both mentioned: 126

 | s_categ     | mean      | std       | nb_elems |
|------------|-----------|-----------|----------|
| accessory  | 3.285714  | 3.638419  | 7        |
| animal     | 0.750000  | 1.832251  | 8        |
| appliance  | 1.444444  | 2.920236  | 9        |
| electronic | -0.714286 | 3.831621  | 14       |
| food       | -1.000000 | NaN       | 1        |
| furniture  | 0.941176  | 2.461468  | 17       |
| indoor     | 0.750000  | 1.164965  | 8        |
| kitchen    | 0.000000  | 0.816497  | 4        |
| outdoor    | 2.000000  | 1.549193  | 10       |
| person     | 4.615385  | 3.571612  | 13       |
| sports     | 0.181818  | 1.328020  | 5       |
| vehicle    | 2.733333  | 3.657145  | 30       |


![Rank res subset 5](https://github.com/Naiina/Saliency-causal-analysis/blob/main/Rank_5.pdf)


## HOI
-hoi_out: 297
-Both not mentioned: 53  /  No hoi only mentioned: 24  /  HOI only mentioned: 74  /  Both mentioned: 146

|       s_categ   |    mean   |    std  |    nb_elems |
|------------|-----------|-----------|----------|
|0    accessory  | 10.000000   |    NaN   |      1|
|1       animal   | 3.307692 | 5.647782   |     13|
|2    appliance |  2.884615  |2.943833    |    26|
|3   electronic |  1.937500  |2.313181    |    32|
|4         food |  4.818182  |3.458849    |    11|
|5    furniture |  2.678571  |2.326236    |    28|
|6       indoor |  1.500000   |2.258318   |      6|
|7      kitchen   4.166667   |3.311596    |     6|
|8      outdoor |  2.222222  | 2.488864   |      9|
|9       sports |  4.500000  | 4.949747   |      2|
|10     vehicle |  1.500000  | 3.261344   |     12|


