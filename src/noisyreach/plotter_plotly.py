import numpy as np
import plotly.graph_objects as go

LINE_WIDTH = 3
LINE_WIDTH_3D = 3


def new_plot(plot3d=False):
    fig = go.Figure()
    init_plot(fig, plot3d=plot3d)
    return fig


def is3d(fig: go.Figure):
    return "scene" in fig.layout


def init_plot(fig: go.Figure, plot3d=False):
    fig.update_layout(
        template="plotly_white",
        font=dict(size=18),
    )
    if plot3d:
        fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8), projection_type="orthographic"
                ),
            ),
        )
    return fig


def plot_trace(
    fig: go.Figure,
    trace: np.ndarray,
    *,
    name=None,
    t_dim=0,
    x_dim=1,
    y_dim=2,
    color=None,
    alpha=1.0,
):
    if is3d(fig):
        fig.add_trace(
            go.Scatter3d(
                x=trace[:, x_dim],
                y=trace[:, y_dim],
                z=trace[:, t_dim],
                mode="lines",
                line=dict(
                    width=LINE_WIDTH_3D,
                    color=color,
                ),
                showlegend=(name is not None),
                name=name,
                opacity=alpha,
            )
        )

        balance_axes(fig)
    else:
        fig.add_trace(
            go.Scatter(
                x=trace[:, x_dim],
                y=trace[:, y_dim],
                mode="markers",
                showlegend=True,
                name=name,
                opacity=alpha,
            )
        )

    return fig


def balance_axes(fig: go.Figure):
    """Make a 3d plot's x and y axes have the same range."""
    if not is3d(fig):
        raise ValueError("balance_axes can only be used with 3D plots")

    # Get data from all traces
    vmin = np.inf
    vmax = -np.inf
    for trace in fig.data:
        if hasattr(trace, "x"):
            vmin = min(vmin, np.min(trace.x))
            vmax = max(vmax, np.max(trace.x))
        if hasattr(trace, "y"):
            vmin = min(vmin, np.min(trace.y))
            vmax = max(vmax, np.max(trace.y))

    # Update layout with equal ranges
    fig.update_layout(
        scene=dict(
            xaxis_range=(vmin, vmax),
            yaxis_range=(vmin, vmax),
        )
    )

    return fig


def plot_safetube(
    fig: go.Figure,
    trace: np.ndarray,
    radius,
    *,
    name="nominal",
    t_dim=0,
    x_dim=1,
    y_dim=2,
):
    if not is3d(fig):
        raise ValueError("plot_safetube3d can only be used with 3D plots")

    n_theta = 20  # number of points around each circle
    theta = np.linspace(0, 2 * np.pi, n_theta)

    # Preallocate arrays for the tube coordinates
    circles_t = np.empty((len(trace), n_theta))
    circles_x = np.empty((len(trace), n_theta))
    circles_y = np.empty((len(trace), n_theta))

    # Generate the circle at each point along the line
    for i, row in enumerate(trace):
        circles_t[i, :] = row[t_dim]
        circles_x[i, :] = row[x_dim] + radius * np.cos(theta)
        circles_y[i, :] = row[y_dim] + radius * np.sin(theta)

    fig.add_trace(
        go.Surface(
            x=circles_x,
            y=circles_y,
            z=circles_t,
            name="safe region",
            colorscale=[[0, "lightblue"], [1, "lightblue"]],
            opacity=0.5,
            showscale=False,
            # hoverinfo="skip",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=trace[:, x_dim],
            y=trace[:, y_dim],
            z=trace[:, t_dim],
            name=name,
            mode="lines",
            line=dict(color="black", width=LINE_WIDTH_3D),
            opacity=0.8,
        )
    )

    balance_axes(fig)

    return fig


def display_fig(fig: go.Figure):
    # Remove duplicate legend items
    seen = {}
    for trace in fig.data:
        if trace.name in seen:
            trace.showlegend = False
        else:
            seen[trace.name] = True
    fig.show()


def savefig(fig: go.Figure, filename: str):
    fig.write_image(filename)
