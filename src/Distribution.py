import os
import argparse
import yaml
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import random

random.seed(42)


def get_distrib(params):
    main_root = Path(params["src"])
    distributions = {}
    # traverse all folders
    for root, dirs, files in os.walk(main_root):
        class_name = root.replace(main_root.as_posix(), "")
        class_name = class_name[1:]

        # ignore main root
        if len(class_name) == 0:
            continue

        # NOTE: assumme no nested and i dont need to do name concat!!!
        print(f"{class_name} {len(files)}")
        distributions[class_name] = len(files)

    return distributions


def plot_diagrams(path, distrib):
    df = pd.DataFrame(list(distrib.items()), columns=["Class", "Frequency"])

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=(
            "Histogram of Class Frequencies",
            "Pie Chart of Class Frequencies",
        ),
        column_widths=[0.6, 0.4],
    )

    fig.add_trace(
        go.Bar(x=df["Class"],
               y=df["Frequency"],
               name="Frequency"),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Pie(labels=df["Class"], values=df["Frequency"], name="Frequency"),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text="Class Frequency Visualization: Histogram and Pie Chart"
    )

    fig.write_image(f"{path}/distrib.png")


def get_args():
    root_path = os.path.dirname(__file__)
    default_config_path = f"{root_path}/Distribution.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-id",
        "--id",
        help="experiment id for reports and cache",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input_cfg",
        help="input config",
        type=str,
        default=default_config_path
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="optional report output path",
        type=str,
        default=None,
    )
    return parser.parse_args()


def validate_params(data):
    required_keys = ["src"]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"{missing} is missing in config")


def main():
    args = get_args()
    param_file = args.input_cfg

    params = None
    with open(param_file, "r") as file:
        params = yaml.safe_load(file)

    validate_params(params)
    distrib = get_distrib(params)

    root_path = os.path.dirname(__file__)
    report_out_path = f"{root_path}/../reports/{args.id}/distrib/"
    if args.output_path is not None:
        report_out_path = args.output_path
    report_out_path = os.path.abspath(report_out_path)
    Path(report_out_path).mkdir(parents=True, exist_ok=True)
    plot_diagrams(report_out_path, distrib)

    # print mean of distributions
    sum = 0
    for key in distrib.keys():
        sum += distrib[key]
    print(f"mean: {sum / len(distrib.keys())}")

    print(f"charts plotted at {report_out_path}")


main()
