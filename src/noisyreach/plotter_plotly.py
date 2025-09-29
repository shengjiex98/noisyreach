"""Plotly-based visualization utilities for trajectory analysis.

This module provides functions for creating interactive visualizations of
trajectory data, safety tubes, and reachable sets using Plotly. It supports
both 2D and 3D plotting with automatic axis balancing and customizable styling.

The module includes functions for:
- Creating and initializing Plotly figures
- Plotting trajectory traces and safety regions
- Managing 3D visualization with balanced axes
- Displaying and saving interactive plots

Example:
    Create and display a 3D trajectory plot:

    >>> import numpy as np
    >>> from noisyreach.plotter_plotly import new_plot, plot_trace, display_fig
    >>> fig = new_plot(plot3d=True)
    >>> trace = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])  # t, x, y
    >>> plot_trace(fig, trace, name="trajectory")
    >>> display_fig(fig)
"""

import numpy as np
import plotly.graph_objects as go

LINE_WIDTH = 3
LINE_WIDTH_3D = 3


def new_plot(plot3d: bool = False) -> go.Figure:
    """Create a new Plotly figure with default styling.

    Args:
        plot3d: Whether to create a 3D plot (default: False)

    Returns:
        Initialized Plotly figure with appropriate layout
    """
    fig = go.Figure()
    init_plot(fig, plot3d=plot3d)
    return fig


def is3d(fig: go.Figure) -> bool:
    """Check if a Plotly figure is configured for 3D plotting.

    Args:
        fig: Plotly figure to check

    Returns:
        True if figure has 3D scene configuration, False otherwise
    """
    return "scene" in fig.layout


def init_plot(fig: go.Figure, plot3d: bool = False) -> go.Figure:
    """Initialize Plotly figure with default styling and layout.

    Args:
        fig: Plotly figure to initialize
        plot3d: Whether to configure for 3D plotting (default: False)

    Returns:
        Figure with updated layout and styling
    """
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
    name: str | None = None,
    t_dim: int = 0,
    x_dim: int = 1,
    y_dim: int = 2,
    color: str | None = None,
    alpha: float = 1.0,
) -> go.Figure:
    """Add trajectory trace to Plotly figure.

    Plots trajectory data as either 3D lines (for time-series) or 2D markers.
    Automatically detects plot type based on figure configuration.

    Args:
        fig: Plotly figure to add trace to
        trace: Trajectory data array (n_points, n_dims)
        name: Legend name for the trace (optional)
        t_dim: Column index for time dimension (default: 0)
        x_dim: Column index for x-coordinate (default: 1)
        y_dim: Column index for y-coordinate (default: 2)
        color: Trace color (optional)
        alpha: Opacity level 0-1 (default: 1.0)

    Returns:
        Updated figure with new trace added

    Note:
        For 3D plots, time is used as the z-axis to show trajectory evolution.
    """
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


def balance_axes(fig: go.Figure) -> go.Figure:
    """Balance x and y axes in 3D plots to have equal ranges.

    Ensures that spatial coordinates have the same scale for proper
    visualization of geometric relationships.

    Args:
        fig: 3D Plotly figure to balance

    Returns:
        Figure with balanced axis ranges

    Raises:
        ValueError: If figure is not configured for 3D plotting
    """
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
    radius: float,
    *,
    name: str = "nominal",
    t_dim: int = 0,
    x_dim: int = 1,
    y_dim: int = 2,
) -> go.Figure:
    """Add safety tube visualization to 3D Plotly figure.

    Creates a cylindrical safety region around a trajectory by plotting
    circular cross-sections at each point along the path.

    Args:
        fig: 3D Plotly figure to add safety tube to
        trace: Nominal trajectory array (n_points, n_dims)
        radius: Safety tube radius in meters
        name: Legend name for nominal trajectory (default: "nominal")
        t_dim: Column index for time dimension (default: 0)
        x_dim: Column index for x-coordinate (default: 1)
        y_dim: Column index for y-coordinate (default: 2)

    Returns:
        Figure with safety tube surface and nominal trajectory

    Raises:
        ValueError: If figure is not configured for 3D plotting

    Note:
        The safety tube is rendered as a semi-transparent surface showing
        the reachable region around the nominal trajectory.
    """
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


def display_fig(fig: go.Figure) -> None:
    """Display Plotly figure with cleaned up legend.

    Removes duplicate legend entries and shows the interactive plot.

    Args:
        fig: Plotly figure to display
    """
    # Remove duplicate legend items
    seen = {}
    for trace in fig.data:
        if trace.name in seen:
            trace.showlegend = False
        else:
            seen[trace.name] = True
    fig.show()


def savefig(fig: go.Figure, filename: str) -> None:
    """Save Plotly figure to file.

    Args:
        fig: Plotly figure to save
        filename: Output filename with extension (e.g., 'plot.png', 'plot.html')

    Note:
        Supported formats depend on installed dependencies (kaleido for images).
    """
    fig.write_image(filename)
