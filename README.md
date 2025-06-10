# No Three in Line Problem Solver

A solver for the "no three in line" problem using Integer Linear Programming (ILP) with Gurobi.

## Requirements

- Python 3.8+ or [uv](https://github.com/astral-sh/uv)
- [Gurobi Optimizer](https://www.gurobi.com/)

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager and project manager written in Rust.

#### Install uv
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

#### Run the project with uv
```bash
# Clone the repository
git clone https://github.com/pranav-ramanathan/no-three-in-line.git
cd no-three-in-line

uv venv --python=3.11
uv sync
uv run main.py --n 17
```

### Manual Installation

If you prefer to set up the environment yourself:

```bash
# Clone the repository
git clone https://github.com/pranav-ramanathan/no-three-in-line.git
cd no-three-in-line

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install gurobipy>=12.0.2 typer>=0.16.0

# Run the solver
python main.py --n 14
```

#### Project Dependencies

- **gurobipy**: Gurobi optimization solver
- **typer**: CLI framework for the command-line interface
- **Python>=3.11**: Required Python version

## Usage

### Basic Usage

```bash
# Solve for a 14×14 grid
python main.py --n 14

# Solve for a 17×17 grid  
python main.py --n 17

# Using uv
uv run main.py --n 19
```

## How It Works

### Overview

1. **Grid Generation**: Creates all points in the n×n grid
2. **Line Detection**: Finds all lines containing 3 or more points
3. **Model Formulation**: 
   - **Variables**: Binary variable for each grid point (selected/not selected)
   - **Objective**: Maximize number of selected points
   - **Constraints**: At most 2 points per line with 3+ collinear points
4. **Optimization**: Uses Gurobi
5. **Solution Display**: Shows optimal point placement