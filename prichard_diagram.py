"""Plotting code accompanying <paper>."""

import json
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors
import numpy as np
import pandas as pd
from typing import Literal
import warnings

from helper_functions import format_sig_figs

HardwareInfo = dict[str, dict[Literal["marker", "cores", "cost_per_node_per_hour"], int | str]]


def read_data(data_path: str, hardware_info_path: str, timestep_size: float) -> tuple[pd.DataFrame, HardwareInfo]:
    hardware_info: HardwareInfo
    with open(hardware_info_path) as hardware_info_file:
        hardware_info = json.loads(hardware_info_file.read())

    data = pd.read_csv(data_path)
    data["normalized_runtime"] = data["elapsed_time"] / (timestep_size * data["timestep_count"])
    for machine_type in data["machine_type"].unique():
        if machine_type not in hardware_info:
            raise RuntimeError(f"Machine type {machine_type} not found in hardware info: `{hardware_info_path}`")
        hardware_cost = float(hardware_info[machine_type]["cost_per_node_per_hour"])

        slice = data["machine_type"] == machine_type
        data.loc[slice, "normalized_cost"] = (
            hardware_cost / 3600 * data.loc[slice, "node_count"] * data.loc[slice, "normalized_runtime"]
        )
        data.loc[slice, "subscription"] = data.loc[slice, "total_cores"] / data.loc[slice, "node_count"]
        hardware_cores = int(hardware_info[machine_type]["cores"])
        data.loc[slice, "undersubscription"] = hardware_cores - data.loc[slice, "subscription"]
    return data, hardware_info


def remove_suboptimal_undersubscription(
    data: pd.DataFrame,
    x_column: str = "normalized_runtime",
    y_column: str = "normalized_cost",
) -> pd.DataFrame:
    mask = [True for _ in range(data.shape[0])]
    for i, row in enumerate(data.iterrows()):
        row_x_suboptimality = data[x_column] < row[1][x_column]
        row_y_suboptimality = data[y_column] < row[1][y_column]
        if True in (row_x_suboptimality & row_y_suboptimality).value_counts():
            mask[i] = False
    return data.loc[mask].reset_index()


#
# def remove_suboptimal_undersubscription(data: pd.DataFrame) -> pd.DataFrame:
#     """Returns a copy of the DataFrame containing only the fastest
#     undersubscription at each node count/hardware combination.
#
#     Parameters
#     ----------
#     1. data : DataFrame
#             - The complete dataset.
#
#     Returns
#     -------
#     - DataFrame
#         - The trimmed dataset.
#     """
#     optimal_rows = None
#     for hardware in data["machine_type"].unique():
#         hardware_rows = data["machine_type"] == hardware
#         for i, row in data[hardware_rows].iterrows():
#             hardware_nodes_rows = hardware_rows & (data["node_count"] == row["node_count"])
#             if row["normalized_cost"] > min(data.loc[hardware_nodes_rows, "normalized_cost"]):
#                 hardware_rows[i] = False
#         if optimal_rows is None:
#             optimal_rows = hardware_rows
#         else:
#             optimal_rows |= hardware_rows
#     return data[optimal_rows]


def prichard_diagram(
    data: pd.DataFrame,
    marker_types: dict[str, str] | None = None,
    ax: Axes | None = None,
    show_suboptimal: bool = True,
    plot_kwargs: dict = {},
    make_legend: bool = True,
    show_pareto_front: bool = True,
    label_optima: bool = True,
    fill_pareto_region: bool = True,
    machine_types: str | list[str] | None = None,
    max_free_cores: int | None = None,
    undersubscription_count: int | None = None,
) -> tuple[Figure, Axes]:

    assert isinstance(data, pd.DataFrame)

    if not show_pareto_front and label_optima:
        warnings.warn("Optima will not be labeled if `show_pareto_front` is False")

    if ax is None:
        fig, ax = plt.subplots()
        assert isinstance(ax, Axes)
        ax.set_xlabel("Normalized Runtime [s/flow-s]")
        ax.set_ylabel("Normalized Cost [$/flow-s]")
        ax.set_xscale("log")
        ax.set_yscale("log")
    elif isinstance(ax, Axes):
        fig = ax.figure  # type: ignore[assignment]

    if isinstance(max_free_cores, int):
        data = data[data["undersubscription"] <= max_free_cores].reset_index()

    if not show_suboptimal:
        data = remove_suboptimal_undersubscription(data)

    if machine_types is not None:
        machine_type_slice = pd.Series(False, index=data.index)
        if type(machine_types) is str:
            machine_type_slice = data["machine_type"] == machine_types
        else:
            for machine_name in machine_types:
                machine_type_slice |= data["machine_type"] == machine_name
        data = data[machine_type_slice].reset_index()

    if len(data) == 0:
        raise RuntimeError("No data found.")

    for hardware_name in data["machine_type"].unique():
        hardware_rows = data["machine_type"] == hardware_name
        x_data = data.loc[hardware_rows, "normalized_runtime"]
        y_data = data.loc[hardware_rows, "normalized_cost"]
        if isinstance(marker_types, dict):
            plot_kwargs["marker"] = marker_types[hardware_name]
        ax.scatter(
            x_data,
            y_data,
            label=hardware_name,
            edgecolors="black",
            alpha=0.8,
            **plot_kwargs,
        )

    if show_pareto_front or fill_pareto_region:
        pareto_set = [
            list(x) for x in remove_suboptimal_undersubscription(data)[["normalized_runtime", "normalized_cost"]].values
        ]
        pareto_set.sort()

        pareto_indices = None
        for x, y in pareto_set:
            slice = np.logical_and(
                data["normalized_runtime"] == x,
                data["normalized_cost"] == y,
            )
            if pareto_indices is None:
                pareto_indices = slice
            else:
                pareto_indices = np.logical_or(pareto_indices, slice)

        pareto_set.insert(0, [pareto_set[0][0], 10**9])
        pareto_set.append([10**9, pareto_set[-1][-1]])
        # Transposes from a list of rows to a list of columns
        pareto_set = list(zip(*pareto_set))  # type: ignore[arg-type]

        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        if show_pareto_front:
            ax.plot(
                *pareto_set,
                color=plot_kwargs["color"] if "color" in plot_kwargs else "black",
                label="Pareto front",
            )
            if label_optima:
                label_pareto_front(ax, data[pareto_indices])
        if fill_pareto_region:
            ax.fill(
                [*pareto_set[0], pareto_set[0][-1]],
                [*pareto_set[1], pareto_set[1][0]],
                fill=True,
                color="#00000022",
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if make_legend:
        ax.legend()

    if label_optima and show_pareto_front:
        ax.set_xlim(left=ax.get_xlim()[0]*0.6)
        ax.set_ylim(bottom=ax.get_ylim()[0]*0.6)
    return fig, ax


def label_pareto_front(
    ax: Axes,
    pareto_data: pd.DataFrame,
    label_offset: tuple[float, float] = (-90, -40),
    show_values: bool = False,
    color: str | None = None,
) -> list[matplotlib.text.Annotation]:
    """Adds data labels to the pareto front of a scatterplot

    Parameters
    ----------
    1. ax : Axes
            - The axes to which data labels should be added
    2. pareto_data : DataFrame
            - The XY data to be labeled
    3. label_offset : tuple[float,float], (default [10, 50])
            - The distance each label should be offset from the
              corresponding data point
    """

    def get_hardware_color(axis: Axes, hardware_name: str) -> str | np.ndarray:
        """Searches an axes for a particular hardware's series. Returns a
        muted version of that series's color.

        Parameters
        ----------
        1. axis : Axes
                - The axes containing the hardware's series.
        2. hardware_name : str
                - The name of the hardware

        Returns
        -------
        - tuple[float, float, float]
            - A tuple containing RGB values of the color.
        """
        color: str | np.ndarray = "#c8c8c8"  # Default color that will be used if hardware_name is not in axis labels
        for child in axis.get_children():
            if type(child) is matplotlib.collections.PathCollection and child.get_label() == hardware_name:
                color = np.array(child.get_facecolor()[0])
                color = color * 0.7
                color = np.array([min(c, 1) for c in color])
                color[-1] = 1  # Set alpha to 1?
                break
        return color

    text_handles = []
    for _, row in pareto_data.sort_values("normalized_runtime").iterrows():
        x = row["normalized_runtime"]  # X variable value
        y = row["normalized_cost"]  # Y variable value
        x_pixels, y_pixels = ax.transData.transform((x, y))  # X and Y coordinates (in pixels) of this row's datapoint
        pixels_to_coords = (
            ax.transData.inverted().transform
        )  # A transform function to translate pixels into figure coordinates
        label = f"{row['node_count']:.0f}×{row['subscription']:.0f}"
        if show_values:
            label += f"\n({format_sig_figs(x, 3)}s, ${format_sig_figs(y, 3)})"
        text_handles.append(
            ax.annotate(
                label,
                xy=(x, y),
                xytext=pixels_to_coords((x_pixels + label_offset[0], y_pixels + label_offset[1])),  # type: ignore[arg-type]
                arrowprops=dict(width=1, headlength=8, headwidth=6, shrink=0.1, color="#111111"),
                multialignment="center",
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(
                    boxstyle="round",
                    fc="#c8c8c8cc",
                    ec=get_hardware_color(ax, row["machine_type"]) if color is None else color,
                ),
            )
        )

    return text_handles


def make_prichard_plot(
    data_path: str, hardware_info_path: str, timestep_size: float, show_figure: bool = True
) -> tuple[Figure, Axes]:
    data, machine_info = read_data(data_path, hardware_info_path, timestep_size)
    marker_types: dict[str, str] = {machine_name: machine_info[machine_name]["marker"] for machine_name in machine_info}  # type: ignore[misc]
    fig, ax = prichard_diagram(data, marker_types)
    if show_figure:
        plt.show()
    return fig, ax
