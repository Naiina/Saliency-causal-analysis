import csv
import tqdm
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter,defaultdict
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
import spacy
nlp = spacy.load("en_core_web_sm")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr,pearsonr
import json




import nltk
from nltk.corpus import wordnet as wn

# Ensure WordNet is available
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Original COCO-like category dict
base_categories = {
    'person': ['person'],
    'vehicle': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'], 
    'outdoor': ['traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'],
    'animal': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'], 
    'accessory': ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase'], 
    'sports': ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
    'kitchen': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'], 
    'food': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
             'hot dog', 'pizza', 'donut', 'cake'],
    'furniture': ['chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet'],
    'electronic': ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'],
    'appliance': ['microwave', 'oven', 'toaster', 'sink', 'refrigerator'],
    'indoor': ['book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
}






nlp = spacy.load("en_core_web_sm")

def save_feat():

    # --- 1) LOAD SYNONYMS FROM JSON ---
    with open("coco_synonyms.json", "r") as f:
        d_categ = json.load(f)

    l_mentioned = []
    l_deprel = []
    l_rank = []
    l_img_id = []
    l_size = []
    l_caption = []
    l_category = []

    with open("changed_5_filtered_captions.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            img = row[1]
            categ = img.split(".")[0].split("_")[1]

            # Identify size from filename
            size = "small" if "small" in img else "big"

            caption = row[2][2:-2]
            doc = nlp(caption)

            # --- 2) GET SYNONYMS FOR THIS CATEGORY ---
            # fallback: if category not found in JSON, use itself
            l_categ = d_categ.get(categ, [categ])

            # Also compare against lemmatized forms
            l_categ_lemmas = set([nlp(w)[0].lemma_ for w in l_categ])

            order = 0
            found = False

            # --- 3) FIND FIRST MATCHING NOUN (SINGULAR OR PLURAL) ---
            for token in doc:
                if token.pos_ == "NOUN":
                    order += 1

                    # normalize the token
                    lemma = token.lemma_.lower()

                    if lemma in l_categ_lemmas:
                        found = True
                        deprel = token.dep_
                        rank = order
                        break

            # --- 4) SAVE RESULTS ---
            if not found:
                #print("############## NOT FOUND")
                #print("Category:", categ)
                #print("Synonyms:", l_categ)
                #print("Caption:", caption)

                l_mentioned.append(0)
                l_deprel.append("none")
                l_rank.append(-1)
            else:
                l_mentioned.append(1)
                l_deprel.append(deprel)
                l_rank.append(rank)

            l_img_id.append(img.split("_")[0])
            l_size.append(size)
            l_caption.append(caption)
            l_category.append(categ)


    d = {
        "mentioned": l_mentioned,
        "deprel": l_deprel,
        "rank": l_rank,
        "img_id": l_img_id,
        "size": l_size,
        "caption": l_caption,
        "categ": l_category
    }

    #for k, v in d.items():
    #    print(k, len(v))

    df = pd.DataFrame.from_dict(d)
    df.to_csv("img_and_captions_changed_5.csv", index=False)




def rank_barplot():

    # Compute means and standard deviations
    means = [np.mean(d_word_order["big"]), np.mean(d_word_order["small"])]
    stds = [np.std(d_word_order["big"]), np.std(d_word_order["small"])]

    # Bar labels
    labels = ['Big', 'Small']

    # Create bar plot
    plt.bar(labels, means, yerr=stds, capsize=8, color=['skyblue', 'salmon'], alpha=0.8)

    # Add labels and title
    plt.ylabel('Average Rank')


    # Optionally add numerical values above bars
    for i, v in enumerate(means):
        plt.text(i, v + stds[i] + 0.1, f"{v:.2f}", ha='center', fontsize=10)

    plt.show()

def deprel_histo():

    # Count occurrences for each group
    count_big = Counter(d_deprel['big'])
    count_small = Counter(d_deprel['small'])

    # Get the set of all dependency relations (keys)
    all_deprels = sorted(set(count_big.keys()) | set(count_small.keys()))

    # Create aligned counts
    big_counts = [count_big.get(dep, 0) for dep in all_deprels]
    small_counts = [count_small.get(dep, 0) for dep in all_deprels]

    # Bar settings
    x = np.arange(len(all_deprels))
    width = 0.35  # width of bars

    # Plot bars side by side
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, big_counts, width, label='Big', color='skyblue')
    bars2 = ax.bar(x + width/2, small_counts, width, label='Small', color='salmon')

    # Labels and formatting
    ax.set_xlabel('Dependency Relation (deprel)')
    ax.set_ylabel('Count')
    ax.set_title('Counts of Dependency Relations for Big vs Small')
    ax.set_xticks(x)
    ax.set_xticklabels(all_deprels, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()



def rank_diff_per_categ():
    df = pd.read_csv("img_and_captions_changed_5.csv")
    print(len(df))

    # Keep only mentioned items
    df = df[df['mentioned'] == 1]
    print(len(df))
    # Pivot or merge to find both big/small for same img_id + categ
    merged = (
        df.pivot_table(index=['img_id', 'categ'], columns='size', values='rank')
        .reset_index()
        .dropna(subset=['big', 'small'])  # keep only those with both sizes
    )
    print(merged)
    #exit()
    # Compute rank difference (e.g., small - big)
    merged['rank_diff'] = merged['small'] - merged['big']

    # Aggregate per category: mean and std
    stats = (
        merged.groupby('categ')['rank_diff']
        .agg(['mean', 'std'])
        .reset_index()
    )

    # 🧹 1. Remove categories with missing or zero std
    stats = stats.dropna(subset=['std'])
    stats = stats[stats['std'] > 0]

    # 📊 2. Sort by average rank difference
    stats = stats.sort_values(by='mean', ascending=False).reset_index(drop=True)

    print(stats)

    # 🎨 3. Plot
    plt.figure(figsize=(8, 5))
    plt.bar(stats['categ'], stats['mean'], yerr=stats['std'],
            capsize=8, color='skyblue', alpha=0.8)

    plt.ylabel('Average Rank Difference (small - big)')
    plt.xlabel('Category')
    plt.title('Average Rank Difference per Category (± Std)')
    plt.xticks(rotation=45, ha='right')

    # Add numerical values above bars
    for i, (m, s) in enumerate(zip(stats['mean'], stats['std'])):
        plt.text(i, m + s + 0.1, f"{m:.2f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()



def rank_bar_per_categ():


    df = pd.read_csv("img_and_captions.csv")

    stats = (
    df[df['mentioned'] == 1]
    .groupby(['categ', 'size'])['rank']
    .agg(['mean', 'std'])
    .reset_index()
    )

    # Remove categories with NaN or 0 std if needed
    stats = stats.dropna(subset=['std'])
    stats = stats[stats['std'] > 0]

    # Pivot so we have columns for "big" and "small"
    pivot = stats.pivot(index='categ', columns='size', values='mean')
    pivot_std = stats.pivot(index='categ', columns='size', values='std')

    # Sort categories by average of both sizes (for nice ordering)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    pivot_std = pivot_std.loc[pivot.index]

    # Define bar positions
    x = np.arange(len(pivot.index))
    width = 0.35

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bars for big and small
    bars_big = ax.bar(x - width/2, pivot['big'], width, yerr=pivot_std['big'],
                    capsize=8, label='Big', color='skyblue', alpha=0.8)
    bars_small = ax.bar(x + width/2, pivot['small'], width, yerr=pivot_std['small'],
                        capsize=8, label='Small', color='salmon', alpha=0.8)

    # Labels and aesthetics
    ax.set_ylabel('Average Rank')
    ax.set_xlabel('Category')
    ax.set_title('Average Rank per Category (Big vs Small)')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.legend()

    # Optional: Add value labels
    for i, cat in enumerate(pivot.index):
        ax.text(x[i] - width/2, pivot['big'][i] + 0.1, f"{pivot['big'][i]:.2f}", ha='center', fontsize=9)
        ax.text(x[i] + width/2, pivot['small'][i] + 0.1, f"{pivot['small'][i]:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

def deprel_heatmap():
    
    df = pd.read_csv("img_and_captions.csv")
    keep_deprels = ["nsubjpass", "pobj", "nsubj", "dobj"]
    df = df[df["deprel"].isin(keep_deprels)]

    # Pivot so we can match big/small per img_id
    pivot_df = df.pivot(index="img_id", columns="size", values="deprel").dropna()

    # Count combinations
    counts = pivot_df.value_counts().reset_index(name="count")
    
    keep_deprels = ["nsubjpass", "pobj", "nsubj", "dobj"]
    # Create a pivot table for the heatmap
    heatmap_data = counts.pivot(index="big", columns="small", values="count").fillna(0)

    order = ["nsubj", "nsubjpass", "dobj", "pobj"]
    heatmap_data = heatmap_data.reindex(index=order, columns=order, fill_value=0)

    # Plot
    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Big vs Small Deprel Co-occurrence")
    plt.xlabel("Small Deprel")
    plt.ylabel("Big Deprel")
    plt.show()




def filter_size_rank():
    df_bbox = pd.read_csv("COCO/val2017/coco_modif_05_11_segmented/bbox_annot.csv")
    df_rank = pd.read_csv("img_and_captions.csv")

    df_rank = df_rank.rename(columns={"size": "type"})

    df_bbox = df_bbox.copy()
    df_bbox["img_id"] = df_bbox["img"].str.extract(r"(\d+)")  # extract numeric image id
    df_bbox["type"] = df_bbox["img"].apply(lambda x: "small" if "small" in x else "big")
    df_bbox = df_bbox[["img_id","type","size"]]


    df_rank = df_rank.copy()
    df_rank["img_id"] = df_rank["img_id"].astype(str).str.zfill(12)
    df_rank = df_rank[["img_id","type","rank"]]

    print(df_rank.head())
    print(df_bbox.head())
    merged = pd.merge(df_rank, df_bbox, on=["img_id", "type"], how="inner")#, suffixes=("_bbox", "_caption"))
    #merged.to_csv("merged_rank_size.csv", index=False)

    type_counts = merged.groupby(["img_id", "type"]).size().unstack(fill_value=0)

    valid_ids = type_counts[(type_counts.get("big", 0) == 1) & (type_counts.get("small", 0) == 1)].index

    merged = merged[merged["img_id"].isin(valid_ids)].copy()
    merged.to_csv("merged_rank_size.csv")
    
    pivot_df = merged.pivot(index="img_id", columns="type", values=["rank", "size"])

    # Flatten multi-level columns
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]

    # Drop any rows missing big/small pairs
    pivot_df = pivot_df.dropna(subset=["rank_big", "rank_small", "size_big", "size_small"])

    # Filter out rows with invalid rank (-1)
    pivot_df = pivot_df[(pivot_df["rank_big"] != -1) & (pivot_df["rank_small"] != -1)]

    # Compute differences
    pivot_df["rank_diff"] = pivot_df["rank_big"] - pivot_df["rank_small"]
    pivot_df["size_diff"] = pivot_df["size_big"] - pivot_df["size_small"]

    # Merge back to original df if needed
    df_with_diff = merged.merge(pivot_df[["rank_diff", "size_diff"]], on="img_id", how="left")

    # Save result
    df_with_diff.to_csv("merged_with_diffs.csv", index=False)
    print(df_with_diff.head())





def correlation_size_rank():
    
    df = pd.read_csv("merged_with_diffs.csv")
    df = df[["rank_diff","size_diff"]]
    df.drop_duplicates(inplace=True)
    print(df)
    df.dropna(subset=["rank_diff", "size_diff"],inplace=True)

    print("Data size:", df.shape)
    print("Unique rank_diff values:", df["rank_diff"].unique())
    print("Unique size_diff values:", df["size_diff"].unique())
 

    corr, pval = pearsonr(df["rank_diff"], df["size_diff"])
    print(f"Pearson correlation: {corr:.3f}, p-value: {pval:.3e}")

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="rank_diff", y="size_diff")

    slope, intercept = np.polyfit(df["rank_diff"], df["size_diff"], 1)
    x_vals = np.array(plt.gca().get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, linestyle="--", color="red", label="Fit line")

    plt.title(f"Rank Difference vs Size Difference\nSpearman r={corr:.3f}")
    plt.xlabel("Rank Difference (big - small)")
    plt.ylabel("Size Difference (big - small)")
    plt.legend()
    plt.grid(True)
    plt.show()

#correlation_size_rank()
#save_feat()

rank_diff_per_categ()