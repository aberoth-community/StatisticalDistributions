import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def printPercents(counts, catTotal, labels):
    if catTotal == 0:
        print("No counts to summarize.")
    else:
        for label, cnt in zip(labels, counts):
            pct = cnt / catTotal * 100.0
            print(f"{label}: {int(cnt)} ({pct:.3f}% of category total)")


# path = "C:\\Users\\devib\\Desktop\\AberothDamageCollection\\30Def\\20Accu\\14Atk.csv"
# path = "C:\\Users\\devib\\Desktop\\AberothDamageCollection\\30Def\\20Accu\\3Atk10DagPlusAccu.csv"
# path = "WeaponDamage\\20Def\\14Atk\\10Accu.csv"
# path = "C:\\Users\\devib\\Desktop\\AberothDamageCollection\\20Def\\14Atk\\10Accu.csv"
# path = "C:\\Users\\devib\\Desktop\\AberothDamageCollection\\15Def\\14Atk\\20Accu.csv"
# path = "DrainLifeSpell\\DrainSpell.csv"
path = "FacetedStone\\FacesFull.csv"

max_xticks = 100
groupTitle = "Damage Distribution (All vs. Hit Types)" # MASTER TITLE (won't overwrite any subplot titles)
title = groupTitle
# Load CSV
df = pd.read_csv(path)

# If column not present, try to guess
col = "Damage Dealt"
if col not in df.columns:
    lower = {c.lower(): c for c in df.columns}
    if "damage" in lower:
        col = lower["damage"]
    elif "damage dealt" in lower:
        col = lower["damage dealt"]
    elif "face" in lower:
        col = lower["face"]
    else:
        candidates = [c for c in df.columns if "dam" in c.lower()]
        if candidates:
            col = candidates[0]
        else:
            raise SystemExit(f'Could not find a "{col}" column (or similar). '
                                f"Columns found: {list(df.columns)}")

# Extract series and support numeric or categorical
orig = df[col].dropna()
if orig.empty:
    raise SystemExit(f'Column "{col}" contains no non-NA values.')

s_num = pd.to_numeric(orig, errors="coerce")

plt.figure()

if s_num.notna().any():
    # For measuring hit damage 
    s_all = s_num.dropna()

    min_val = int(np.floor(s_all.min()))
    max_val = int(np.ceil(s_all.max()))
    n_bins = int(max_val - min_val + 1)
    bin_edges = np.linspace(min_val - 0.5, max_val + 0.5, n_bins + 1)
    labels = np.arange(min_val, max_val + 1)

    # 2x2 grid: [All, Normal, Solid, Crit], sharing x-axis
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.15)
    axes = axes.ravel()

    # Master title (won't overwrite any subplot titles)
    fig.suptitle(groupTitle, fontsize=14, fontweight="bold", y=0.98)

    # get the type of hit the damage hit was
    type_col = "Hit Type"
    type_series = (
        df[type_col].astype(str).str.strip().str.lower()
        if type_col in df.columns else pd.Series(index=df.index, dtype=str)
    )

    subsets = [
        ("All Hit Types", s_all),
        ("Normal Hits Only", s_num[type_series == "normal"].dropna()),
        ("Solid Hits",  s_num[type_series == "solid"].dropna()),
        ("Critical Hits",   s_num[type_series == "crit"].dropna()),
    ]
    
        # Console: percent of total for each bin/category 
    total = 0
    num_norm = 0
    num_solid = 0
    num_crit = 0
    # Draw each histogram using the same bins/edges and style
    for ax, (title, s_sub) in zip(axes, subsets):
        if s_sub.empty:
            ax.set_title(f"{title} (no data)")
            ax.set_xlim(bin_edges[0], bin_edges[-1])
            ax.set_ylabel("Count")
            continue

        counts, edges, patches = ax.hist(
            s_sub.values,
            bins=bin_edges,
            density=False,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.2,
        )
        
        ax.set_title(title)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylabel("Count")

        match title:
            case "All Hit Types":
                total = sum(counts)
                print(f"\nAll Damage Types ({total} total hits):")
                printPercents(counts, total, labels)
            case "Normal Hits Only":
                num_norm = sum(counts)
                print("\nNormal Hits Only")
                printPercents(counts, num_norm, labels)
                print(f"\nNormal Hits as a percentage of total hits ({(num_norm/total) * 100:.3f}% of total hits)")
                printPercents(counts, total, labels)
            case "Solid Hits":
                num_solid = sum(counts)
                print("\nSolid Hits Only")
                printPercents(counts, num_solid, labels)
                print(f"\nSolid Hits as a percentage of total hits ({(num_solid/total) * 100:.3f}% of total hits)")
                printPercents(counts, total, labels)
            case "Critical Hits":
                num_crit = sum(counts)
                print("\nCritical Hits Only")
                printPercents(counts, num_crit, labels)
                print(f"\nCritical Hits as a percentage of total hits ({(num_crit/total) * 100:.3f}% of total hits)")
                printPercents(counts, total, labels)
    
    # Integer x-ticks applied to every subplot
    all_ticks = labels
    if len(all_ticks) > max_xticks:
        step = int(math.ceil(len(all_ticks) / max_xticks))
        ticks = all_ticks[::step]
    else:
        ticks = all_ticks

    for ax in axes:
        ax.set_xticks(ticks)
        ax.tick_params(axis='x', which='both', labelbottom=True)

    # label just one (or all) x-axes
    axes[-1].set_xlabel("Value")
    
else:
    # For measuring the occurance of string values
    s_cat = orig.astype(str)
    vc = s_cat.value_counts()   # counts by category
    # vc = vc.sort_index()       # alphabetical; comment out to keep "most frequent first"

    labels = vc.index.to_list()
    counts = vc.values.astype(float)

    x = np.arange(len(labels))
    plt.bar(x, counts, alpha=0.7, edgecolor="black", linewidth=1.2)
    plt.title(title)
    # Tick labels
    if len(labels) > max_xticks:
        step = int(math.ceil(len(labels) / max_xticks))
        sel = np.arange(0, len(labels), step)
        plt.xticks(sel, [labels[i] for i in sel], rotation=45, ha="right")
    else:
        plt.xticks(x, labels, rotation=45, ha="right")
        
    # Log percent of total for each bin/category
    total = counts.sum()
    if total == 0:
        print("No counts to summarize.")
    else:
        print(f"\nTotal Throws: {total}\n")
        # For numeric labels (np.ndarray), ensure Python ints for pretty printing
        pretty_labels = [int(x) if isinstance(x, (np.integer, np.floating)) and float(x).is_integer() else x
                        for x in labels]
        for label, cnt in zip(pretty_labels, counts):
            pct = cnt / total * 100.0
            print(f"{label}: {int(cnt)} ({pct:.3f}% of total)")

# average line, greatest deviation line, legend 
avg_per_bin = float(np.mean(counts))
greatest_dev = 0.0
greatest_dev_bin_val = 0.0
for val in counts:
    dev = abs(val - avg_per_bin)
    if dev > greatest_dev:
        greatest_dev = dev
        greatest_dev_bin_val = val


# Indicate which bar has a greatest deviation from the average number of counts
# plt.axhline(avg_per_bin, linestyle="--", linewidth=1.5, color="black", alpha=0.9,
#             label=f"Avg count/bin = {avg_per_bin:.2f}")
# plt.axhline(greatest_dev_bin_val, linestyle="--", linewidth=1.5, color="red", alpha=0.9,
#             label=f"Greatest Deviation = {greatest_dev:.2f}")
# plt.legend()

plt.xlabel(col)
plt.ylabel("Count")
plt.tight_layout()
plt.show()
...

