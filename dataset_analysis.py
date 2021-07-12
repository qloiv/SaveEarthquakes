# load train test and dev dataset
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, gaussian_kde
from scipy.stats import spearmanr


def analyse(cp):
    catalog = pd.read_csv(cp)

    train = catalog[catalog["SPLIT"] == "TRAIN"]
    test = catalog[catalog["SPLIT"] == "TEST"]
    val = catalog[catalog["SPLIT"] == "DEV"]

    num_train = len(train)
    num_test = len(test)
    num_val = len(val)
    num_dataset = len(catalog)

    print(num_train, num_train / num_dataset)
    print(num_test, num_test / num_dataset)
    print(num_val, num_val / num_dataset)

    events = sorted(catalog["EVENT"].unique())
    events_train = sorted(train["EVENT"].unique())
    events_test = sorted(test["EVENT"].unique())
    events_val = sorted(val["EVENT"].unique())
    print(len(events), len(events_train), len(events_test), len(events_val))

    stations = sorted(catalog["STATION"].unique())
    stations_train = sorted(train["STATION"].unique())
    stations_test = sorted(test["STATION"].unique())
    stations_val = sorted(val["STATION"].unique())
    print(len(stations), len(stations_train), len(stations_test), len(stations_val))

    depth = catalog["DEPTH"]
    depth_train = train["DEPTH"]
    depth_test = test["DEPTH"]
    depth_val = val["DEPTH"]
    print(
        depth.max(),
        depth.min(),
        depth_train.max(),
        depth_train.min(),
        depth_test.max(),
        depth_train.min(),
        depth_val.max(),
        depth_val.min(),
    )

    distances = catalog["DIST"]
    distances_train = train["DIST"]
    distances_test = test["DIST"]
    distances_val = val["DIST"]
    print(
        distances.max(),
        distances.min(),
        distances_train.max(),
        distances_train.min(),
        distances_test.max(),
        distances_train.min(),
        distances_val.max(),
        distances_val.min(),
    )
    # hypo or epicentral distance

    print(np.all(distances > depth))
    print(np.any(np.greater(depth, distances)) == True)

    # p to s time
    s_picks = catalog["S_PICK"]
    p_picks = catalog["P_PICK"]
    diff = np.array(s_picks - p_picks)
    diff.sort()
    t = np.linspace(start=0, stop=1, num=len(diff))

    fig_sp = plt.figure()
    axes = fig_sp.add_subplot(111)
    fig_sp.suptitle(
        "Time between P-Pick and S-Pick. Up to "
        + str(np.round(np.sum(diff <= 20) / len(diff), decimals=2))
        + "% of the examples\n(from all train, test or dev sets) may include the S-Wave"
    )
    plt.scatter(diff, t, s=0.5)
    axes.axvline(20, color="black", linestyle="dashed", linewidth=0.5)
    plt.xlabel("P to S-wave time in seconds")
    plt.ylabel("Fraction of Examples")
    fig_sp.savefig("P to S Time.png")

    magnitudes = catalog["MA"]
    magnitudes_train = train["MA"]
    magnitudes_test = test["MA"]
    magnitudes_val = val["MA"]
    print(
        magnitudes.max(),
        magnitudes.min(),
        magnitudes_train.max(),
        magnitudes_train.min(),
        magnitudes_test.max(),
        magnitudes_train.min(),
        magnitudes_val.max(),
        magnitudes_val.min(),
    )
    print(np.cov(magnitudes, distances))
    print(pearsonr(magnitudes, distances))
    print(spearmanr(magnitudes, distances))

    # Compute pie slices
    train_array = np.array(magnitudes_train)
    test_array = np.array(magnitudes_test)
    val_array = np.array(magnitudes_val)
    bars = np.concatenate((np.concatenate((train_array, test_array)), val_array))
    bars_sorted = np.sort(bars)
    bars_argsorted = np.argsort(bars)
    colours = np.array(
        ["b"] * len(train_array) + ["r"] * len(test_array) + ["g"] * len(val_array)
    )
    colours_sorted = colours[bars_argsorted]
    x = np.arange(0, len(bars), 1)

    N = len(bars)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = bars_sorted
    width = 2 * np.pi / N - 0.001
    colors = colours

    # fig1 = plt.figure()
    # ax = fig1.add_subplot(111, projection='polar')
    # ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

    # fig1.savefig("polarPlot.png")

    # fig1 = plt.figure()
    # ax = fig1.add_subplot(111, projection='polar')
    # ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

    # fig1.savefig("polarPlot.png")e

    train_array = np.array(magnitudes_train)

    test_array = np.array(magnitudes_test)
    val_array = np.array(magnitudes_val)
    bars = np.concatenate((np.concatenate((train_array, test_array)), val_array))
    bars_sorted = np.sort(bars)
    bars_argsorted = np.argsort(bars)
    colours = np.array(
        ["b"] * len(train_array) + ["r"] * len(test_array) + ["g"] * len(val_array)
    )
    colours_sorted = colours[bars_argsorted]
    x = np.arange(0, len(bars), 1)

    fig2 = plt.figure()
    axes = fig2.add_subplot(111)
    # axes.bar(x, bars, width = 0.8,align = "edge", bottom=0.0, color=colours)
    sns.histplot(magnitudes, bins=100)
    axes.set_yscale("log")
    plt.xlabel("Magnitude")
    plt.ylabel("Number of Examples")
    fig2.savefig("histogram_ma.png")

    fig5 = plt.figure()
    axes = fig5.add_subplot(111)
    # axes.bar(x, bars, width = 0.8,align = "edge", bottom=0.0, color=colours)
    sns.histplot(distances / 1000, bins=100)
    plt.xlabel("Distance in km")
    plt.ylabel("Number of Examples")
    fig5.savefig("histogram_dist.png")

    fig6 = plt.figure()
    axes = fig6.add_subplot(111)
    # axes.bar(x, bars, width = 0.8,align = "edge", bottom=0.0, color=colours)
    sns.histplot(depth, bins=100)
    plt.xlabel("Depth in km")
    plt.ylabel("Number of Examples")
    fig6.savefig("histogram_depth.png")

    fig7 = plt.figure()
    fig7.suptitle("Distance to Depth Scatterplot")
    axes = fig7.add_subplot(111)
    # axes.bar(x, bars, width = 0.8,align = "edge", bottom=0.0, color=colours)
    x = distances / 1000
    y = depth
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    cm = plt.cm.get_cmap("plasma")
    x, y, z = x[idx], y[idx], z[idx]
    axes.scatter(x, y, c=z, cmap=cm, s=1, alpha=0.05)
    plt.xlabel("Distance[km]", fontsize=8)
    plt.ylabel("Depth[km]", fontsize=8)
    fig7.savefig("depth_distance.png", dpi=600)

    # fig3 = plt.figure()
    # axes1 = fig3.add_subplot(111)

    # axes1.scatter(magnitudes, distances, s=1)
    # fig3.savefig("scatterplot.png")

    # print(np.cov(magnitudes, distances))
    # print(pearsonr(magnitudes, distances))
    # print(spearmanr(magnitudes, distances))

    #
    # fig3 = plt.figure()
    # axes1 = fig3.add_subplot(111)


#    axes1.scatter(magnitudes, distances / 1000, s=0.2)
#    axes1.set_xscale("log")
#    fig3.savefig("scatterplot.png")

# magnitudes.plot()
# plt.show()


# analyse(cp="/home/viola/WS2021/Code/Daten/Chile_small/new_catalog.csv")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--catalog_path", type=str)
    args = parser.parse_args()
    action = args.action

    if action == "analyse":
        analyse(cp=args.catalog_path)
