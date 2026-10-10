#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/thesis_figures/dataset_figures"
    / "paper_positive_gradcam_negative_3panel.png"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive-image",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--positive-overlay",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--negative-image",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def load_image(path):
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    return plt.imread(path)


def main():
    args = parse_args()

    positive_image = load_image(args.positive_image)
    positive_overlay = load_image(args.positive_overlay)
    negative_image = load_image(args.negative_image)

    figure = plt.figure(figsize=(13.5, 5.3))
    grid = figure.add_gridspec(
        1,
        3,
        left=0.035,
        right=0.975,
        bottom=0.07,
        top=0.79,
        wspace=0.10,
    )

    positive_axis = figure.add_subplot(grid[0, 0])
    overlay_axis = figure.add_subplot(grid[0, 1])
    negative_axis = figure.add_subplot(grid[0, 2])

    panels = [
        (positive_axis, positive_image, "(a) Original CXR"),
        (overlay_axis, positive_overlay, "(b) Grad-CAM Overlay"),
        (negative_axis, negative_image, "(c) Original CXR"),
    ]

    for axis, image, title in panels:
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(
            title,
            fontsize=12,
            fontweight="bold",
            pad=7,
        )
        axis.axis("off")

    figure.text(
        0.50,
        0.965,
        "Representative Test-Set Cases",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    figure.canvas.draw()

    positive_position = positive_axis.get_position()
    overlay_position = overlay_axis.get_position()
    negative_position = negative_axis.get_position()

    figure.text(
        (positive_position.x0 + overlay_position.x1) / 2,
        0.885,
        "Positive case",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    figure.text(
        (negative_position.x0 + negative_position.x1) / 2,
        0.885,
        "Negative case",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    separator_x = (
        overlay_position.x1
        + negative_position.x0
    ) / 2

    figure.add_artist(
        Line2D(
            [separator_x, separator_x],
            [0.09, 0.86],
            transform=figure.transFigure,
            linewidth=2.2,
            alpha=0.9,
            color="black",
        )
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
