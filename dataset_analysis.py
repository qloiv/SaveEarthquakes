# load train test and dev dataset
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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

    events = sorted(catalog['EVENT'].unique())
    events_train = sorted(train["EVENT"].unique())
    events_test = sorted(test["EVENT"].unique())
    events_val = sorted(val["EVENT"].unique())
    print(len(events), len(events_train), len(events_test), len(events_val))

    # stations = sorted(catalog["STATION"].unique())
    # stations_train = sorted(train["STATION"].unique())
    # stations_test = sorted(test["STATION"].unique())
    # stations_val = sorted(val["STATION"].unique())
    # print(len(stations),len(stations_train),len(stations_test),len(stations_val))

    distances = (catalog['DIST'])
    distances_train = train["DIST"]
    distances_test = test["DIST"]
    distances_val = val["DIST"]
    print(distances.max(), distances.min(), distances_train.max(), distances_train.min(), distances_test.max(),
          distances_train.min(),
          distances_val.max(), distances_val.min())

    magnitudes = (catalog['MA'])
    magnitudes_train = train["MA"]
    magnitudes_test = test["MA"]
    magnitudes_val = val["MA"]
    print(magnitudes.max(), magnitudes.min(), magnitudes_train.max(), magnitudes_train.min(), magnitudes_test.max()
          , magnitudes_train.min(),
          magnitudes_val.max(), magnitudes_val.min())

    # Compute pie slices
    train_array = np.array(magnitudes_train)
    test_array = np.array(magnitudes_test)
    val_array = np.array(magnitudes_val)
    bars = np.concatenate((np.concatenate((train_array, test_array)), val_array))
    bars_sorted = np.sort(bars)
    bars_argsorted = np.argsort(bars)
    colours = np.array(["b"] * len(train_array) + ["r"] * len(test_array) + ["g"] * len(val_array))
    colours_sorted = colours[bars_argsorted]
    x = np.arange(0, len(bars), 1)

    N = len(bars)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = bars_sorted
    width = 2 * np.pi / N - 0.001
    colors = colours

    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='polar')
    ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

    fig1.savefig("polarPlot.png")

    train_array = np.array(magnitudes_train)

    test_array = np.array(magnitudes_test)
    val_array = np.array(magnitudes_val)
    bars = np.concatenate((np.concatenate((train_array, test_array)), val_array))
    bars_sorted = np.sort(bars)
    bars_argsorted = np.argsort(bars)
    colours = np.array(["b"] * len(train_array) + ["r"] * len(test_array) + ["g"] * len(val_array))
    colours_sorted = colours[bars_argsorted]
    x = np.arange(0, len(bars), 1)

    fig2 = plt.figure()
    axes = fig2.add_subplot(111)
    # axes.bar(x, bars, width = 0.8,align = "edge", bottom=0.0, color=colours)
    axes.hist([train_array, test_array, val_array], alpha=1, bins=len(bars), histtype="stepfilled")
    fig2.savefig("histogram.png")

    fig3 = plt.figure()
    axes1 = fig3.add_subplot(111)

    axes1.scatter(magnitudes, distances, s=1)
    fig3.savefig("scatterplot.png")

    print(np.cov(magnitudes, distances))
    print(pearsonr(magnitudes, distances))
    print(spearmanr(magnitudes, distances))

    magnitudes.plot()
    plt.show()


analyse(cp="/home/viola/WS2021/Code/Daten/Chile_small/new_catalog.csv"
        )
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, required=True)
    parser.add_argument('--catalog_path', type=str)
    args = parser.parse_args()
    action = args.action

    if action == 'analyse':
        analyse(cp=args.catalog_path)
