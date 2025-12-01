import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns




def rank_diff_per_categ_size(csv):
    df = pd.read_csv(csv)
    print("all data: ", len(df))

    df = df[df['mentioned'] == 1]
    print("mentioned: ", len(df))
    # Pivot or merge to find both big/small for same img_id + categ
    merged = (
        df.pivot_table(index=['img_id', 's_categ'], columns='size', values='rank')
        .reset_index()
        .dropna(subset=['big', 'small'])
    )

    merged['rank_diff'] = merged['small'] - merged['big']
    print("pairs: ", len(merged))
 
    stats = (
    merged.groupby('s_categ')['rank_diff']
    .agg(['mean', 'std', 'count'])  # add count
    .reset_index()
    .rename(columns={'count': 'nb_elems'})  # rename column
    )
    print(stats)
    stats = stats.dropna(subset=['std'])
    stats = stats[stats['std'] > 0]

    stats = stats.sort_values(by='mean', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.bar(stats['s_categ'], stats['mean'], yerr=stats['std'],
            capsize=8, color='skyblue', alpha=0.8)

    plt.ylabel('Average Rank Difference (small - big)')
    plt.xlabel('Category')
    plt.title('Average Rank Difference per Category (± Std)')
    plt.xticks(rotation=45, ha='right')

    for i, (m, s) in enumerate(zip(stats['mean'], stats['std'])):
        plt.text(i, m + s + 0.1, f"{m:.2f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig("Rank_5.pdf")
    plt.show()




def rank_diff_per_categ_hoi(csv):
    df = pd.read_csv(csv)
    print("all data: ", len(df))

    df = df[df['mentioned'] == 1]
    print("mentioned: ", len(df))
    # Pivot or merge to find both big/small for same img_id + categ
    merged = (
        df.pivot_table(index=['img_id', 's_categ'], columns='hoi', values='rank')
        .reset_index()
        .dropna(subset=['hoi', 'no_hoi'])
    )

    merged['rank_diff'] = merged['no_hoi'] - merged['hoi']
    print("pairs: ", len(merged))
 
    stats = (
    merged.groupby('s_categ')['rank_diff']
    .agg(['mean', 'std', 'count'])  # add count
    .reset_index()
    .rename(columns={'count': 'nb_elems'})  # rename column
    )
    print(stats)
    stats = stats.dropna(subset=['std'])
    stats = stats[stats['std'] > 0]

    stats = stats.sort_values(by='mean', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.bar(stats['s_categ'], stats['mean'], yerr=stats['std'],
            capsize=8, color='skyblue', alpha=0.8)

    plt.ylabel('Average Rank Difference (no_hoi - hoi)')
    plt.xlabel('Category')
    plt.title('Average Rank Difference per Category (± Std)')
    plt.xticks(rotation=45, ha='right')

    for i, (m, s) in enumerate(zip(stats['mean'], stats['std'])):
        plt.text(i, m + s + 0.1, f"{m:.2f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig("Rank_hoi.pdf")
    plt.show()




feat = "hoi"

if feat == "size":
    csv = "COCO/val2017/img_and_captions_changed_temp.csv"
    rank_diff_per_categ_size(csv)
if feat == "hoi":
    csv = "COCO/val2014/img_and_captions_hoi_detect_standing_h.csv"
    rank_diff_per_categ_hoi(csv)



