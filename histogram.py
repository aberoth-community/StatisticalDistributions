# Note: Much of the code in this program is made from code generated or derived from code generated AI. Use accordingly.
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


defense = 10
attack = 14
accuracy = 10
path = f"WeaponDamage\\{defense}Def\\{attack}Atk\\{accuracy}Accu.csv"

# path = "WeaponDamage\\30Def\\20Accu\\3Atk10DagPlusAccu.csv"
# path = "WeaponDamage\\30Def\\3Atk\\10Accu\\3Atk_MH_only_accu_off_hand2.csv"
# path = "DrainLifeSpell\\DrainSpell.csv"
path = "FacetedStone\\FacesFull.csv"

max_xticks = 100
groupTitle = f"Damage Distributions {defense} Def, {attack} Atk, {accuracy} Accu" # MASTER TITLE (won't overwrite any subplot titles)

def main(): 
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

            cat_dist_percents = []
            match title:
                case "All Hit Types":
                    total = sum(counts)
                    print(f"\nAll Damage Types ({total} total hits):")
                    cat_dist_percents = print_percents(counts, total, labels)
                    fit = fit_and_plot_normal(ax, edges, cat_dist_percents, line_color="orange")
                    if fit is not None:
                        mu_hat, sigma_hat = fit
                        print(f"Fitted Normal for '{title}': mu={mu_hat:.6f}, sigma={sigma_hat:.6f}")

                case "Normal Hits Only":
                    num_norm = sum(counts)
                    print("\nNormal Hits Only")
                    cat_dist_percents = print_percents(counts, num_norm, labels)
                    fit = fit_and_plot_normal(ax, edges, cat_dist_percents, line_color="orange")
                    if fit is not None:
                        mu_hat, sigma_hat = fit
                        print(f"Fitted Normal for '{title}': mu={mu_hat:.6f}, sigma={sigma_hat:.6f}")
                        
                    print(f"\nNormal Hits as a percentage of total hits ({(num_norm/total) * 100:.3f}% of total hits)")
                    print_percents(counts, total, labels)
                case "Solid Hits":
                    num_solid = sum(counts)
                    print("\nSolid Hits Only")
                    cat_dist_percents = print_percents(counts, num_solid, labels)
                    fit = fit_and_plot_normal(ax, edges, cat_dist_percents, line_color="orange")
                    if fit is not None:
                        mu_hat, sigma_hat = fit
                        print(f"Fitted Normal for '{title}': mu={mu_hat:.6f}, sigma={sigma_hat:.6f}")
                    
                    print(f"\nSolid Hits as a percentage of total hits ({(num_solid/total) * 100:.3f}% of total hits)")
                    print_percents(counts, total, labels)
                case "Critical Hits":
                    num_crit = sum(counts)
                    print("\nCritical Hits Only")
                    cat_dist_percents = print_percents(counts, num_crit, labels)
                    fit = fit_and_plot_normal(ax, edges, cat_dist_percents, line_color="orange")
                    if fit is not None:
                        mu_hat, sigma_hat = fit
                        print(f"Fitted Normal for '{title}': mu={mu_hat:.6f}, sigma={sigma_hat:.6f}")
                        
                    print(f"\nCritical Hits as a percentage of total hits ({(num_crit/total) * 100:.3f}% of total hits)")
                    print_percents(counts, total, labels)
        

        
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
                print(f"{label}: {int(cnt)} ({pct:.4f}% of total)")

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

def print_percents(counts, catTotal, labels):
    percents = []
    if catTotal == 0:
        print("No counts to summarize.")
    else:
        for label, cnt in zip(labels, counts):
            pct = cnt / catTotal * 100.0
            print(f"{label}: {int(cnt)} ({pct:.4f}% of category total)")
            percents.append((label, cnt))
    return percents


def fit_and_plot_normal(ax, edges, label_count_pairs, line_color="orange", min_label_offset = 0.5):
    """Fit Normal(μ,σ) to binned counts using interval likelihood (infinite tails for
    first/last bins), then overlay the scaled curve."""
    import numpy as np, math

    if not label_count_pairs:
        return None

    e = np.asarray(edges, dtype=float)             # len = nbins+1, half-integer edges
    bw = (e[1] - e[0]) if len(e) > 1 else 1.0

    # labels are integer bin centers; counts are raw counts (not percents)
    labels = np.array([int(x) for x, _ in label_count_pairs], dtype=int)
    counts = np.array([float(c) for _, c in label_count_pairs], dtype=float)
    tot = counts.sum()
    if tot <= 0:
        return None

    # Map each label to its bin index k so bin = (e[k], e[k+1]]
    min_label = int(round(e[0] + min_label_offset))             # since edges start at min-0.5
    k = labels - min_label
    lo = e[k].astype(float)
    hi = e[k + 1].astype(float)

    # Extend tails to ±∞ so we account for mass outside the displayed range
    lo[k == 0] = -np.inf
    hi[k == (len(e) - 2)] = np.inf

    # Initial guess from center-of-bin moments (just to seed the search)
    xc = labels.astype(float)
    w = counts
    mu0 = (xc * w).sum() / tot
    ex2 = (xc * xc * w).sum() / tot
    var0 = max(ex2 - mu0 * mu0, (bw * 0.5) ** 2, 1e-3)
    sigma0 = math.sqrt(var0)
    rng = max(e[-1] - e[0], 1.0)

    def Phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def bin_prob(mu, sg, l, h):
        if sg <= 1e-9:
            return 0.0
        if math.isinf(l) and l < 0:
            return Phi((h - mu) / sg)
        if math.isinf(h) and h > 0:
            return 1.0 - Phi((l - mu) / sg)
        return Phi((h - mu) / sg) - Phi((l - mu) / sg)

    def loglike(mu, sg):
        if sg <= 1e-9:
            return -1e300
        ll = 0.0
        for l, h, c in zip(lo, hi, counts):
            pk = bin_prob(mu, sg, l, h)
            if pk <= 0.0:
                return -1e300
            ll += c * math.log(pk)
        return ll

    # Coarse + refine grid (no SciPy), widened so μ can sit far left of data
    best_ll, best_mu, best_sg = -1e300, mu0, sigma0
    M, S = 141, 101
    mu_grid = np.linspace(mu0 - 6 * rng, mu0 + 6 * rng, M)
    sg_grid = np.linspace(max(0.1 * bw, sigma0 / 8), max(sigma0 * 6, bw * 8), S)
    for mu in mu_grid:
        for sg in sg_grid:
            ll = loglike(mu, sg)
            if ll > best_ll:
                best_ll, best_mu, best_sg = ll, mu, sg

    mu2 = np.linspace(best_mu - min_label * rng, best_mu + min_label * rng, 101) # mu2 = np.linspace(best_mu - 1.5 * rng, best_mu + 1.5 * rng, 101)
    sg2 = np.linspace(max(bw * 0.05, best_sg / 3), best_sg * 3, 101)
    for mu in mu2:
        for sg in sg2:
            ll = loglike(mu, sg)
            if ll > best_ll:
                best_ll, best_mu, best_sg = ll, mu, sg

    mu, sigma = float(best_mu), float(best_sg)

    # Overlay curve on counts scale
    xx = np.linspace(e[0], e[-1], 400)
    pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xx - mu) / sigma) ** 2)
    yy = tot * bw * pdf
    ax.plot(xx, yy, color=line_color, linewidth=2,
            label=f"Mean: {mu:.3f}, STD: {sigma:.3f})")
    ax.legend()
    return mu, sigma



if __name__ == "__main__":
    main()