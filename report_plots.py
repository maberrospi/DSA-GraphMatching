from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import numpy as np


def plot_bars(exp_type):
    assert exp_type in ["features", "nbs", "ft_norm", "ft_maps"]

    # Neighbors
    data = pd.DataFrame(
        {
            "mean": [46.1, 47.5, 45.8, 44.3],
            "lb": [42.7, 43.4, 41.8, 39.6],
            "up": [49.3, 51.0, 49.7, 47.3],
            "nbs": [3, 5, 7, 9],
        }
    )

    # Features used
    data1 = pd.DataFrame(
        {
            "mean": [47.5, 51.3, 47.3, 51.3],
            "lb": [43.4, 47.4, 43.3, 47.4],
            "up": [51.0, 54.3, 50.1, 54.3],
            "features": ["LM", "LM+Coordinates", "LM+Radius", "LM+Coordinates+Radius"],
        }
    )

    # Feature Normalization
    data2 = pd.DataFrame(
        {
            "mean": [47.5, 47.4, 43.5, 46.3, 40.1],
            "lb": [43.4, 42.2, 38.2, 42.0, 34.5],
            "up": [51.0, 51.4, 48.0, 49.9, 45.6],
            "ft_norm": [
                "None",
                "Z-score (Instance)",
                "Min-Max (Instance)",
                "Z-score (Distribution)",
                "Min-Max (Distribution)",
            ],
        }
    )

    data3 = pd.DataFrame(
        {
            "mean": [9.7, 47.5],
            "lb": [7.3, 43.4],
            "up": [11.7, 51.0],
            "ft_maps": ["Temporal UNet", "LM Filter Bank"],
        }
    )

    # fmt:off
    match exp_type:
        case 'features':
            ax = sns.barplot(
            data=data1, x="features", y="mean", hue="features", legend=False, palette="colorblind"
            )
            ax.set_xlabel("Features Used")
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=10)
            ax.set_yticks(range(30, int(max(data1['up']))+3, 2))
            ax.set_ylim(30)
            plt.tight_layout()
            yerr = [data1["mean"] - data1["lb"], data1["up"] - data1["mean"]]
            y_data = data1['mean']
        case 'nbs':
            ax = sns.barplot(
            data=data, x="nbs", y="mean", hue="nbs", legend=False, palette="colorblind"
            )
            ax.set_xlabel("# Neighbors")
            ax.set_yticks(range(30, 53, 2))
            ax.set_ylim(30)
            yerr = [data["mean"] - data["lb"], data["up"] - data["mean"]]
            y_data = data["mean"]
        case 'ft_norm':
            ax = sns.barplot(
            data=data2, x="ft_norm", y="mean", hue="ft_norm", legend=False, palette="colorblind"
            )
            ax.set_xlabel("Normalization Type")
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=10)
            ax.set_yticks(range(30, 53, 2))
            ax.set_ylim(30)
            plt.tight_layout()
            yerr = [data2["mean"] - data2["lb"], data2["up"] - data2["mean"]]
            y_data = data2['mean']
        case 'ft_maps':
            ax = sns.barplot(
            data=data3, x="ft_maps", y="mean", hue="ft_maps", legend=False, palette="colorblind"
            )
            ax.set_xlabel("Feature Maps")
            yerr = [data3["mean"] - data3["lb"], data3["up"] - data3["mean"]]
            y_data = data3["mean"]
    # fmt:on

    ax.set_ylabel("Recall (%)")
    for container in ax.containers:
        ax.bar_label(container, label_type="center", fmt="%.1f%%")

    plt.errorbar(
        x=range(len(y_data)),
        y=y_data,
        yerr=yerr,
        fmt="none",
        solid_capstyle="butt",
        capsize=4,
        ecolor="dimgray",
    )
    plt.show()


def plot_lines(match_type):

    assert match_type in ["patched", "single", "multi"]

    ft_used = np.repeat(["LM", "LM+Coordinates", "LM+Rads", "LM+Coordinates+Rads"], 5)
    ft_norms = np.tile(
        [
            "None",
            "Z-score (Instance)",
            "Min-Max (Instance)",
            "Z-score (Distribution)",
            "Min-Max (Distribution)",
        ],
        4,
    )

    # Additional Results 1
    # fmt:off
    r_patched = [47.4,47.1,43.1,46.1,39.6,51.5,50.4,47.3,46.9,42.1,47.3,47.3,43.9,46.0,40.1,51.5,51.2,48.5,47.2,43.0]
    r_single = [42.3,41.4,35.2,40.0,29.4,43.3,47.7,43.6,46.2,41.4,42.3,42.9,35.5,43.8,30.1,43.4,47.9,44.7,45.9,41.6]
    r_multi = [45.6,45.3,34.9,43.3,30.3,40.3,49.5,41.3,49.4,40.3,45.3,45.9,35.5,45.3,30.2,40.6,48.7,43.1,49.1,40.4]

    # Additional Results 2
    # r_patched = [47.4,47.1,43.1,46.1,39.6,47.4,47.2,47.3,46.2,47.1,47.3,47.3,43.9,46.2,40.1,47.3,48.2,48.5,46.2,47.6]
    # r_single = [42.3,41.4,35.2,40.0,29.4,42.3,42.6,43.6,45.1,41.6,42.3,42.9,35.5,43.8,30.1,42.3,42.9,44.7,45.2,41.6]
    # r_multi = [45.6,45.3,34.9,43.3,30.3,45.6,45.4,41.3,43.5,40.4,45.3,45.9,35.5,45.3,30.2,45.3,46.1,43.1,45.6,40.4]
    # fmt:on

    data = pd.DataFrame(
        {
            "Features Used": list(ft_used),
            "Feature Normalization": list(ft_norms),
            "patched": r_patched,
            "single": r_single,
            "multi": r_multi,
        }
    )

    match match_type:
        case "patched":
            data_wide = data.pivot(
                index="Features Used", columns="Feature Normalization", values="patched"
            )
            title = "Patched"
        case "single":
            data_wide = data.pivot(
                index="Features Used", columns="Feature Normalization", values="single"
            )
            title = "Single-Scale"
        case "multi":
            data_wide = data.pivot(
                index="Features Used", columns="Feature Normalization", values="multi"
            )
            title = "Multi-Scale"

    # print(data_wide_patched)
    sns.set_style("darkgrid")
    ax = sns.lineplot(data=data_wide)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("Recall (%)")
    ax.set_title("GB-VM-" + title)
    # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=10)
    ax.grid(axis="y")
    plt.tight_layout()
    plt.show()


# fmt: off
def get_args():
    parser = argparse.ArgumentParser(description='Prepare and/or evaluate method using annotations')
    parser.add_argument('--plot_type', '-p', default='bars', help='Type of plot to type: bars or lines.')
    parser.add_argument('--exp_type', '-t', default='features', help='Experiment to plot for bar plot.')
    parser.add_argument('--match_type', '-mt', default='patched', help='Match type to plot experiments for.')
    
    return parser.parse_args()
# fmt: on


def main(plot_type, exp_type, match_type) -> None:
    if plot_type == "bars":
        plot_bars(exp_type=exp_type)
    else:
        plot_lines(match_type=match_type)


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
