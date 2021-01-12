import argparse


def read_paths_from_cli():
    """
    Read command line arguments that point to data.

    :return: (catalog_path, waveform_path, model_path)
    """

    # Default paths:
    catalog_path = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
    waveform_path = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
    model_path = "/home/viola/WS2021/Code/Models"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--catalog", default=catalog_path, help="path to event catalog")

    parser.add_argument(
        "--waveforms", default=waveform_path, help="path to waveforms")

    parser.add_argument(
        "--model", default=model_path, help="path to pytorch models")

    args = parser.parse_args()
    return args.catalog, args.waveforms, args.model
