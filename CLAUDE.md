# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package called `noisyreach` that focuses on noisy reachability analysis for autonomous agents, particularly car simulation and trajectory analysis. The project uses the VERSE library (included as a workspace dependency) for agent modeling, simulation, and verification.

## Commands

### Environment Setup

```bash
# Install dependencies (uses uv package manager)
uv sync

# Install with notebook dependencies
uv sync --group notebook

# Install development dependencies
uv sync --group dev
```

### Code Quality

```bash
# Run linter (through pre-commit)
uv run pre-commit run ruff --all-files

# Run formatter (through pre-commit)
uv run pre-commit run ruff-format --all-files

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

### Testing

```bash
# Run tests (if any exist in the main package - check for test discovery)
uv run python -m pytest

# Run VERSE library tests
cd deps/verse-library && uv run python -m pytest tests/
```

## Architecture

### Core Components

**Main Package (`src/noisyreach/`)**:

- `CarAgent`: Main agent class extending VERSE's BaseAgent for car simulation
- `CarMode`: Decision logic enumeration for car agent states
- `Trajectory`: Trajectory definition and management
- `deviation()`: Core function for deviation analysis
- Plotting modules: `plotter_matplotlib.py` and `plotter_plotly.py` for visualization

**Key Dependencies**:

- **VERSE Library**: Core simulation framework (included as workspace dependency in `deps/`)
- **Control**: Python control systems library for dynamics
- **Scipy/Numpy**: Numerical computation
- **Plotly/Matplotlib**: Visualization

### Project Structure

This project uses a **uv workspace** setup:

- Main package in `src/noisyreach/`
- VERSE library as workspace member in `deps/verse-library/`
- Demo implementations in `demo/`

The VERSE library provides the foundational agent framework, with `noisyreach` extending it for specific car reachability analysis scenarios.

### Code Conventions

- Uses **Ruff** for linting and formatting (configured in pyproject.toml)
- Line length: 88 characters
- Modern Python type hints (list[T], dict[K,V], | None for optionals)
- Pre-commit hooks for code quality enforcement
- Snake_case naming for functions/variables, PascalCase for classes

## Development Notes

- The project targets Python 3.12+
- Uses `uv` as the package manager instead of pip
- VERSE library is included as a workspace dependency, not external
- Demos should be placed in `demo/` directory
- Main package exports: `CarAgent`, `CarMode`, `deviation`, `Trajectory`
