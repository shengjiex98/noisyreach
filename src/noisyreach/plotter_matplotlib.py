"""Matplotlib-based visualization utilities for trajectory analysis.

This module provides functions for creating publication-quality static plots
of trajectory data, safety regions, and reachable sets using Matplotlib.
It supports both 2D and 3D plotting with customizable styling.

The module includes functions for:
- Creating and configuring matplotlib axes
- Plotting trajectory traces and safety tubes
- Managing legends and duplicate entries
- Saving high-quality figures

Example:
    Create a 2D trajectory plot:

    >>> import numpy as np
    >>> from noisyreach.plotter_matplotlib import new_plot, plot_trace, show
    >>> ax = new_plot()
    >>> trace = np.array([[0, 1, 2], [1, 2, 3]])  # t, x, y
    >>> plot_trace(ax, trace, name="trajectory")
    >>> show(ax, xlabel="X position", ylabel="Y position")
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

LINE_WIDTH = 1
LINE_WIDTH_3D = 1


def new_plot(plot3d: bool = False):
    """Create a new matplotlib figure and axis with default styling.

    Args:
        plot3d: Whether to create a 3D plot (default: False)

    Returns:
        Matplotlib axis object (Axes or Axes3D)
    """
    fig = plt.figure(figsize=(6, 4), dpi=200)
    if plot3d:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
    init_plot(ax)
    return ax


def init_plot(ax):
    """Initialize matplotlib axis with default styling and projection.

    Configures the axis for optimal visualization based on plot type.
    For 3D plots, sets orthographic projection and balanced aspect ratio.

    Args:
        ax: Matplotlib axis to initialize (Axes or Axes3D)

    Returns:
        Initialized axis object
    """
    if isinstance(ax, Axes3D):
        ax.set_proj_type("ortho")
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect([1, 1, 1])
    else:
        ax.set_facecolor("white")
    return ax


def plot_trace(
    ax,
    trace: np.ndarray,
    *,
    name: str | None = None,
    x_dim: int = 1,
    y_dim: int = 2,
    t_dim: int = 0,
    color: str | None = None,
    linewidth: float = LINE_WIDTH,
    alpha: float = 1.0,
):
    """Add trajectory trace to matplotlib axis.

    Plots trajectory data as either 3D lines (for time-series) or 2D lines.
    Automatically detects plot type based on axis configuration.

    Args:
        ax: Matplotlib axis to plot on (Axes or Axes3D)
        trace: Trajectory data array (n_points, n_dims)
        name: Legend label for the trace (optional)
        x_dim: Column index for x-coordinate (default: 1)
        y_dim: Column index for y-coordinate (default: 2)
        t_dim: Column index for time dimension (default: 0)
        color: Line color (optional)
        linewidth: Line width in points (default: LINE_WIDTH)
        alpha: Opacity level 0-1 (default: 1.0)

    Returns:
        Updated axis object

    Note:
        For 3D plots, time is used as the z-axis to show trajectory evolution.
    """
    if isinstance(ax, Axes3D):
        ax.plot3D(
            trace[:, x_dim],
            trace[:, y_dim],
            trace[:, t_dim],
            linewidth=LINE_WIDTH_3D,
            color=color,
            label=name,
        )
    else:
        ax.plot(
            trace[:, x_dim],
            trace[:, y_dim],
            linewidth=linewidth,
            color=color,
            label=name,
            alpha=alpha,
        )
    return ax


def plot_safetube(
    ax,
    trace: np.ndarray,
    radius: float,
    *,
    name: str = "safe region",
    t_dim: int = 0,
    x_dim: int = 1,
    y_dim: int = 2,
):
    """Add safety tube visualization to matplotlib axis.

    Creates a safety region around a trajectory. For 3D plots, renders as
    a cylindrical surface. For 2D plots, shows as filled regions and circles.

    Args:
        ax: Matplotlib axis to plot on (Axes or Axes3D)
        trace: Nominal trajectory array (n_points, n_dims)
        radius: Safety tube radius in meters
        name: Legend label for safe region (default: "safe region")
        t_dim: Column index for time dimension (default: 0)
        x_dim: Column index for x-coordinate (default: 1)
        y_dim: Column index for y-coordinate (default: 2)

    Returns:
        Updated axis object

    Note:
        The safety tube represents the reachable region around the
        nominal trajectory, typically derived from reachability analysis.
    """
    if isinstance(ax, Axes3D):
        n_theta = 20
        theta = np.linspace(0, 2 * np.pi, n_theta)

        circles_t = np.empty((len(trace), n_theta))
        circles_x = np.empty((len(trace), n_theta))
        circles_y = np.empty((len(trace), n_theta))

        for i, row in enumerate(trace):
            circles_t[i, :] = row[t_dim]
            circles_x[i, :] = row[x_dim] + radius * np.cos(theta)
            circles_y[i, :] = row[y_dim] + radius * np.sin(theta)

        ax.plot_surface(
            circles_x,
            circles_y,
            circles_t,
            alpha=0.7,
            color="lightblue",
            label=name,
        )
    else:
        ax.fill_between(
            trace[:, x_dim],
            trace[:, y_dim] + radius,
            trace[:, y_dim] - radius,
            color="lightblue",
        )
        for row in trace:
            circle = Circle(
                (row[x_dim], row[y_dim]),
                radius,
                color="lightblue",
                alpha=1.0,
                label=name,
            )
            ax.add_patch(circle)

    return ax


def plot_nominal(
    ax,
    trace: np.ndarray,
    *,
    name: str = "nominal",
    x_dim: int = 1,
    y_dim: int = 2,
    t_dim: int = 0,
    color: str = "black",
):
    """Plot nominal (reference) trajectory with emphasized styling.

    Plots the reference trajectory with thicker lines to distinguish
    it from other traces in the visualization.

    Args:
        ax: Matplotlib axis to plot on
        trace: Nominal trajectory array (n_points, n_dims)
        name: Legend label (default: "nominal")
        x_dim: Column index for x-coordinate (default: 1)
        y_dim: Column index for y-coordinate (default: 2)
        t_dim: Column index for time dimension (default: 0)
        color: Line color (default: "black")

    Returns:
        Updated axis object
    """
    plot_trace(
        ax,
        trace,
        color="black",
        name=name,
        x_dim=x_dim,
        y_dim=y_dim,
        t_dim=t_dim,
        linewidth=2 * LINE_WIDTH,
    )

    return ax


def deduplicate_legend(ax):
    """Remove duplicate entries from matplotlib legend.

    Ensures that each unique label appears only once in the legend,
    which is useful when plotting multiple traces with the same name.

    Args:
        ax: Matplotlib axis with legend to clean up

    Returns:
        Updated axis object with deduplicated legend
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, L) for i, (h, L) in enumerate(zip(handles, labels)) if L not in labels[:i]
    ]
    ax.legend(*zip(*unique), fontsize=14)
    return ax


def show(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Display matplotlib plot with proper formatting.

    Applies final formatting including legend deduplication, axis labels,
    and tight layout before displaying the plot.

    Args:
        ax: Matplotlib axis to display
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
    """
    ax.figure.tight_layout()
    deduplicate_legend(ax)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    plt.show()


def savefig(ax, filename: str) -> None:
    """Save matplotlib figure to file with proper formatting.

    Applies legend deduplication and tight bounding box before saving
    to ensure high-quality output.

    Args:
        ax: Matplotlib axis to save
        filename: Output filename with extension (e.g., 'plot.png', 'plot.pdf')

    Note:
        Supported formats include PNG, PDF, SVG, EPS depending on backend.
    """
    deduplicate_legend(ax)
    ax.figure.savefig(filename, bbox_inches="tight")
