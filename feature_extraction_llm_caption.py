import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
import json
from pycocotools.coco import COCO
import spacy

#from nltk.corpus import wordnet as wn
#import nltk
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger_eng')

#nltk.download('wordnet', quiet=True)
#nltk.download('omw-1.4', quiet=True)



nlp = spacy.load("en_core_web_sm")
def get_dict_categ_to_supercateg(coco):
    cats = coco.loadCats(coco.getCatIds())
    return {cat["name"]:cat["supercategory"] for cat in cats}

def get_effective_dep(token):
    """
    Returns the dependency relation that this token should inherit.
    If it's a conj, climb up until reaching a non-conj head.
    """
    head = token
    i = 0
    while head.dep_ == "conj":
        i+=1
        head = head.head
        print(head)
        print(head.dep_)

    return head.dep_



def save_feat(csv_file,csv_out,feat):
    
    annFile='COCO/annotations/instances_val2017.json'
    coco=COCO(annFile)
    d_categ_to_s_categ = get_dict_categ_to_supercateg(coco)

    with open("coco_synonyms.json", "r") as f:
        d_categ = json.load(f)

    l_mentioned = []
    l_deprel = []
    l_rank = []
    l_img_id = []
    l_feat = []
    l_caption = []
    l_category = []
    l_s_category = []

    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:

            img = row[1]
            categ = img.split(".")[0].split("_")[3]

            size = "small" if "small" in img else "big"
            hoi = "no_hoi" if "human" in img else "hoi"

            caption = row[2][2:-2]
            doc = nlp(caption)

            l_categ = d_categ.get(categ, [categ])
            order = 0
            found = False

            for token in doc:
                
                if token.pos_ == "NOUN":
                    order += 1
                    lemma = token.lemma_.lower()
                    if lemma in l_categ:
                        found = True
                        deprel = get_effective_dep(token)
                        rank = order
                        break

            if not found:
                l_mentioned.append(0)
                l_deprel.append("none")
                l_rank.append(-1)
            else:
                l_mentioned.append(1)
                l_deprel.append(deprel)
                l_rank.append(rank)

            l_img_id.append(img.split("_")[2])
            if feat == "size":
                l_feat.append(size)
            if feat == "hoi":
                l_feat.append(hoi)
            l_caption.append(caption)
            l_category.append(categ)
            l_s_category.append(d_categ_to_s_categ[categ])
    if feat == "hoi":
        d = {
            "mentioned": l_mentioned,
            "deprel": l_deprel,
            "rank": l_rank,
            "img_id": l_img_id,
            "hoi": l_feat,
            "caption": l_caption,
            "categ": l_category,
            "s_categ": l_s_category
        }
    if feat == "size":
        d = {
            "mentioned": l_mentioned,
            "deprel": l_deprel,
            "rank": l_rank,
            "img_id": l_img_id,
            "size": l_feat,
            "caption": l_caption,
            "categ": l_category,
            "s_categ": l_s_category
        }

    df = pd.DataFrame.from_dict(d)
    df.to_csv(csv_out, index=False)




def mention_stats_size(csv_file):

    df = pd.read_csv(csv_file)
    #print(df)
    grouped = df.groupby("img_id")

    both_minus1 = 0
    big_minus1_small_positive = 0
    small_minus1_big_positive = 0
    both_zero = 0

    for img_id, group in grouped:
        big_m = group.loc[group['size']=='big', 'mentioned'].values
        small_m = group.loc[group['size']=='small', 'mentioned'].values

        if len(big_m)==0 or len(small_m)==0:
            continue  

        b = big_m[0]
        s = small_m[0]

        if b==0 and s==0:
            both_minus1 += 1
        elif b==0 and s==1:
            big_minus1_small_positive += 1
        elif s==0 and b==1:
            small_minus1_big_positive += 1
        elif b==1 and s==1:
            both_zero += 1

    print("Both not mentioned:", both_minus1)
    print("Small only mentioned:", big_minus1_small_positive)
    print("Big only mentioned:", small_minus1_big_positive)
    print("Both mentioned:", both_zero)
    mention_heatmap_size(both_minus1, big_minus1_small_positive,small_minus1_big_positive, both_zero)




def mention_stats_hoi(csv):

    df = pd.read_csv(csv)
    #print(df)
    grouped = df.groupby("img_id")

    both_not_mentioned = 0
    hoi_only = 0
    no_hoi_only = 0
    both_mentioned = 0

    for img_id, group in grouped:
        hoi_m = group.loc[group['hoi']=='hoi', 'mentioned'].values
        no_hoi_m = group.loc[group['hoi']=='no_hoi', 'mentioned'].values

        if len(hoi_m)==0 or len(no_hoi_m)==0:
            continue  

        h = hoi_m[0]
        nh = no_hoi_m[0]

        if h==0 and nh==0:
            both_not_mentioned += 1
        elif h==0 and nh==1:
            no_hoi_only += 1
        elif nh==0 and h==1:
            hoi_only += 1
        elif h==1 and nh==1:
            both_mentioned += 1

    print("Both not mentioned:", both_not_mentioned)
    print("No hoi only mentioned:", no_hoi_only)
    print("HOI only mentioned:", hoi_only)
    print("Both mentioned:", both_mentioned)
    mention_heatmap_hoi(both_not_mentioned, no_hoi_only,hoi_only, both_mentioned)





def mention_heatmap_hoi(both_not_mentioned, no_hoi_only,hoi_only, both_mentioned):


    # Your variables
    data = np.array([
        [both_not_mentioned, no_hoi_only],
        [hoi_only, both_mentioned]
    ])

    labels = [
        ["Both not mentioned", "No HOI only"],
        ["HOI only", "Both mentioned"]
    ]

    fig, ax = plt.subplots()
    im = ax.imshow(data)   # uses matplotlib's default colormap

    # Show values on cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, str(data[i, j]), ha="center", va="center")

    # Set tick labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    plt.xlabel("No HOI")
    plt.ylabel("HOI")

    plt.title("Mention Heatmap")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()



def mention_heatmap_size(both_minus1, big_minus1_small_positive,small_minus1_big_positive, both_zero):


    data = np.array([
    [both_minus1, big_minus1_small_positive],
    [small_minus1_big_positive, both_zero]
    ])

    labels = [
        ["Both not mentioned", "Small only mentioned"],
        ["Big only mentioned", "Both mentioned"]
    ]

    fig, ax = plt.subplots()
    #im = ax.imshow(data)  # default matplotlib heatmap
    im = ax.imshow(data, cmap="Blues")

    # Annotate each cell with its value
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, str(data[i, j]), ha="center", va="center", color="black")

    # Axis ticks
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    plt.xlabel("Small")
    plt.ylabel("Big")

    plt.title("Mention Heatmap")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("mention_size.pdf")
    plt.show()



feat = "hoi"

if feat == "size":
    csv_file = "COCO/val2014/changed_temp_filtered_captions.csv"
    csv_out = "COCO/val2017/img_and_captions_changed_temp.csv"

if feat == "hoi":
    csv_file = "COCO/val2014/hoi_detect_captions_standing_h.csv"
    csv_out = "COCO/val2014/img_and_captions_hoi_detect_standing_h.csv"



save_feat(csv_file,csv_out,feat)
mention_stats_hoi(csv_out)

#mention_stats()