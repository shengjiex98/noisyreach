import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

LINE_WIDTH = 1
LINE_WIDTH_3D = 1


def new_plot(plot3d=False):
    fig = plt.figure(figsize=(6, 4), dpi=200)
    if plot3d:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
    init_plot(ax)
    return ax


def init_plot(ax):
    if isinstance(ax, Axes3D):
        ax.set_proj_type("ortho")
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect([1, 1, 1])
    else:
        ax.set_facecolor("white")
    return ax


def plot_trace(
    ax,
    trace,
    *,
    name=None,
    x_dim=1,
    y_dim=2,
    t_dim=0,
    color=None,
    linewidth=LINE_WIDTH,
    alpha=1.0,
):
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


def plot_safetube(ax, trace, radius, *, name="safe region", t_dim=0, x_dim=1, y_dim=2):
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
            circle = plt.Circle(
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
    trace,
    *,
    name="nominal",
    x_dim=1,
    y_dim=2,
    t_dim=0,
    color="black",
):
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
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, L) for i, (h, L) in enumerate(zip(handles, labels)) if L not in labels[:i]
    ]
    ax.legend(*zip(*unique), fontsize=14)
    return ax


def show(ax, xlabel=None, ylabel=None):
    ax.figure.tight_layout()
    deduplicate_legend(ax)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    plt.show()


def savefig(ax, filename):
    deduplicate_legend(ax)
    ax.figure.savefig(filename, bbox_inches="tight")
