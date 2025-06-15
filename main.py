from gurobipy import Model, GRB
from math import gcd
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Callable, Any
from typing_extensions import Annotated
from functools import wraps
import time
from enum import Enum

import typer

Point = Tuple[int, int]
Line = Tuple[int, int, int]  # Represents line equation ax + by + c = 0
PointSet = Set[Point]

app = typer.Typer()


class SymmetryType(str, Enum):
    NONE = "none"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    BOTH_AXES = "both_axes"
    DIAGONAL = "diagonal"
    ANTI_DIAGONAL = "anti_diagonal"
    BOTH_DIAGONALS = "both_diagonals"
    ROTATIONAL_90 = "rotational_90"
    ROTATIONAL_180 = "rotational_180"
    ROTATIONAL_270 = "rotational_270"
    ALL = "all"


def time_it(func: Callable) -> Callable:
    """
    Decorator to time function execution and display in minutes and seconds format.

    Args:
        func: Function to be timed

    Returns:
        Wrapped function with timing functionality
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        elapsed = end - start
        mins, secs = divmod(elapsed, 60)

        if mins > 0:
            print(f"Execution time: {mins}m {secs:.2f}s")
        else:
            print(f"Execution time: {secs:.2f}s")

        return result
    return wrapper


def generate_grid_points(n: int) -> List[Point]:
    """Generate all points in an n*n grid."""
    return [(i, j) for i in range(n) for j in range(n)]


def normalize_line_equation(a: int, b: int, c: int) -> Line:
    """
    Normalize a line equation ax + by + c = 0 to canonical form.

    This ensures that equivalent lines have the same representation by:
    1. Dividing by GCD to get smallest integer coefficients
    2. Ensuring consistent sign (a > 0, or a = 0 and b > 0)

    Args:
        a, b, c: Coefficients of line equation ax + by + c = 0

    Returns:
        Normalized line coefficients as (a, b, c)
    """

    # Reduce coefficients by their GCD
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a, b, c = a // g, b // g, c // g

    # Ensure consistent sign convention
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c

    return (a, b, c)


def get_line_through_points(p1: Point, p2: Point) -> Line:
    """
    Calculate the line equation passing through two points.

    For points (x1, y1) and (x2, y2), the line equation is:
    (y1 - y2)x + (x2 - x1)y + (x1*y2 - y1*x2) = 0

    Args:
        p1, p2: Two distinct points

    Returns:
        Normalized line equation coefficients
    """
    x1, y1 = p1
    x2, y2 = p2

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - y1 * x2

    return normalize_line_equation(a, b, c)


def find_collinear_points(points: List[Point]) -> Dict[Line, PointSet]:
    """
    Group points by the lines they lie on.

    For each pair of points, we calculate the line passing through them
    and add both points to that line's point set.

    Args:
        points: List of all grid points

    Returns:
        Dictionary mapping each line to the set of points lying on it
    """
    lines_to_points = defaultdict(set)

    for p1 in points:
        for p2 in points:
            if p1 != p2:
                line = get_line_through_points(p1, p2)
                lines_to_points[line].add(p1)
                lines_to_points[line].add(p2)

    return dict(lines_to_points)


def get_symmetric_point(point: Point, n: int, symmetry_type: str) -> Point:
    """
    Get the symmetric counterpart of a point for different symmetry types.
    
    Args:
        point: Original point (i, j)
        n: Grid size
        symmetry_type: Type of symmetry transformation
        
    Returns:
        Symmetric point
    """
    i, j = point
    
    if symmetry_type == "horizontal":
        return (i, n - 1 - j)
    elif symmetry_type == "vertical":
        return (n - 1 - i, j)
    elif symmetry_type == "diagonal":
        return (j, i)
    elif symmetry_type == "anti_diagonal":
        return (n - 1 - j, n - 1 - i)
    elif symmetry_type == "rotational_90":
        return (j, n - 1 - i)
    elif symmetry_type == "rotational_180":
        return (n - 1 - i, n - 1 - j)
    elif symmetry_type == "rotational_270":
        return (n - 1 - j, i)
    else:
        raise ValueError(f"Unknown symmetry type: {symmetry_type}")


def get_symmetry_constraints(points: List[Point], n: int, symmetry_type: SymmetryType) -> List[List[Tuple[Point, Point]]]:
    """
    Generate symmetry constraint pairs based on the requested symmetry type.
    
    Args:
        points: All grid points
        n: Grid size
        symmetry_type: Type of symmetry to enforce
        
    Returns:
        List of constraint pairs, where each pair represents points that must have the same selection status
    """
    constraints = []
    
    if symmetry_type == SymmetryType.NONE:
        return constraints
    
    # Define the symmetry transformations to apply
    transformations = []
    
    if symmetry_type in [SymmetryType.HORIZONTAL, SymmetryType.BOTH_AXES, SymmetryType.ALL]:
        transformations.append("horizontal")
    
    if symmetry_type in [SymmetryType.VERTICAL, SymmetryType.BOTH_AXES, SymmetryType.ALL]:
        transformations.append("vertical")
    
    if symmetry_type in [SymmetryType.DIAGONAL, SymmetryType.BOTH_DIAGONALS, SymmetryType.ALL]:
        transformations.append("diagonal")
    
    if symmetry_type in [SymmetryType.ANTI_DIAGONAL, SymmetryType.BOTH_DIAGONALS, SymmetryType.ALL]:
        transformations.append("anti_diagonal")
    
    if symmetry_type in [SymmetryType.ROTATIONAL_90, SymmetryType.ALL]:
        transformations.extend(["rotational_90", "rotational_270"])
    
    if symmetry_type in [SymmetryType.ROTATIONAL_180, SymmetryType.ALL]:
        transformations.append("rotational_180")
    
    # Generate constraint pairs for each transformation
    for transform in transformations:
        constraint_pairs = []
        processed = set()
        
        for point in points:
            if point in processed:
                continue
                
            symmetric_point = get_symmetric_point(point, n, transform)
            
            if symmetric_point in points and point != symmetric_point:
                constraint_pairs.append((point, symmetric_point))
                processed.add(point)
                processed.add(symmetric_point)
        
        if constraint_pairs:
            constraints.append(constraint_pairs)
    
    return constraints


def build_optimization_model(points: List[Point],
                           collinear_groups: Dict[Line, PointSet],
                           min_points: int,
                           symmetry_type: SymmetryType = SymmetryType.NONE,
                           threads: int = None) -> Tuple[Model, Dict[Point, any]]:
    """
    Build the Integer Linear Programming model for the no-three-in-line problem.

    Variables: Binary variable for each grid point (1 = selected, 0 = not selected)
    Objective: Maximize number of selected points
    Constraints:
        - At most 2 points selected on any line with 3+ points
        - At least min_points total points selected (optional tightening)
        - Symmetry constraints if requested

    Args:
        points: All grid points
        collinear_groups: Groups of collinear points
        min_points: Minimum number of points to select
        symmetry_type: Type of symmetry to enforce
        threads: Number of threads to use (None for automatic)

    Returns:
        Tuple of (model, point_variables)
    """
    model = Model("No_Three_In_Line")
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output
    
    # Set thread count if specified
    if threads is not None:
        model.setParam("Threads", threads)
        print(f"Using {threads} threads")
    
    # For problems with symmetry, these parameters can help
    if symmetry_type != SymmetryType.NONE:
        model.setParam("Symmetry", 2)  # Aggressive symmetry detection
        model.setParam("MIPFocus", 1)  # Focus on finding feasible solutions quickly

    # Create binary variables for each point
    point_vars = {
        point: model.addVar(vtype=GRB.BINARY, name=f"select_{point[0]}_{point[1]}")
        for point in points
    }

    # Objective: maximize number of selected points
    model.setObjective(sum(point_vars.values()), GRB.MAXIMIZE)

    # Constraint: at most 2 points on any line
    for line_points in collinear_groups.values():
        if len(line_points) > 2:  # Only constrain lines with 3+ points
            model.addConstr(
                sum(point_vars[p] for p in line_points) <= 2,
                name=f"no_three_on_line"
            )

    # Optional constraint: require at least min_points (helps pruning)
    if min_points > 0:
        model.addConstr(
            sum(point_vars.values()) >= min_points,
            name="minimum_points"
        )

    # Add symmetry constraints
    if symmetry_type != SymmetryType.NONE:
        n = int(len(points) ** 0.5)  # Assuming square grid
        symmetry_constraints = get_symmetry_constraints(points, n, symmetry_type)
        
        constraint_count = 0
        for constraint_group in symmetry_constraints:
            for point1, point2 in constraint_group:
                model.addConstr(
                    point_vars[point1] == point_vars[point2],
                    name=f"symmetry_{constraint_count}"
                )
                constraint_count += 1
        
        print(f"Added {constraint_count} symmetry constraints for {symmetry_type.value} symmetry")

    return model, point_vars


@time_it
def solve_optimization_model(model: Model) -> None:
    """
    Solve the optimization model with timing.

    Args:
        model: The Gurobi optimization model to solve
    """
    print("Solving optimization model...")
    model.optimize()


def check_solution_symmetry(n: int, point_vars: Dict[Point, any]) -> Dict[str, bool]:
    """
    Check what symmetries the solution has.
    
    Args:
        n: Grid size
        point_vars: Point variables from optimization
        
    Returns:
        Dictionary indicating which symmetries the solution has
    """
    selected_points = {point for point, var in point_vars.items() if var.x > 0.5}
    
    symmetries = {}
    
    # Check horizontal symmetry
    horizontal_symmetric = True
    for i, j in selected_points:
        symmetric_point = (i, n - 1 - j)
        if symmetric_point not in selected_points:
            horizontal_symmetric = False
            break
    symmetries['horizontal'] = horizontal_symmetric
    
    # Check vertical symmetry
    vertical_symmetric = True
    for i, j in selected_points:
        symmetric_point = (n - 1 - i, j)
        if symmetric_point not in selected_points:
            vertical_symmetric = False
            break
    symmetries['vertical'] = vertical_symmetric
    
    # Check diagonal symmetry
    diagonal_symmetric = True
    for i, j in selected_points:
        symmetric_point = (j, i)
        if symmetric_point not in selected_points:
            diagonal_symmetric = False
            break
    symmetries['diagonal'] = diagonal_symmetric
    
    # Check anti-diagonal symmetry
    anti_diagonal_symmetric = True
    for i, j in selected_points:
        symmetric_point = (n - 1 - j, n - 1 - i)
        if symmetric_point not in selected_points:
            anti_diagonal_symmetric = False
            break
    symmetries['anti_diagonal'] = anti_diagonal_symmetric
    
    # Check 180-degree rotational symmetry
    rotational_180_symmetric = True
    for i, j in selected_points:
        symmetric_point = (n - 1 - i, n - 1 - j)
        if symmetric_point not in selected_points:
            rotational_180_symmetric = False
            break
    symmetries['rotational_180'] = rotational_180_symmetric
    
    return symmetries


def display_solution(n: int, point_vars: Dict[Point, any], objective_value: float, 
                    show_symmetry_analysis: bool = True):
    """
    Display the optimal solution as a grid.

    Args:
        n: Grid size
        point_vars: Dictionary of point variables from optimization
        objective_value: Optimal objective value
        show_symmetry_analysis: Whether to analyze and display symmetries
    """
    print(f"Maximum number of points selected: {int(objective_value)}")
    print(f"This achieves the theoretical maximum of 2n = {2*n} points")

    # Print grid with 'O' for selected points, '.' for unselected
    for i in range(n):
        row = []
        for j in range(n):
            if point_vars[(i, j)].x > 0.5:  # Point is selected
                row.append("O")
            else:
                row.append(".")
        print(" ".join(row))
    
    if show_symmetry_analysis:
        print("\nSymmetry Analysis:")
        symmetries = check_solution_symmetry(n, point_vars)
        for sym_type, has_symmetry in symmetries.items():
            status = "✓" if has_symmetry else "✗"
            print(f"  {status} {sym_type.replace('_', ' ').title()}")


def save_solution_to_file(n: int, point_vars: Dict[Point, any], objective_value: float, 
                         filename: str = None, symmetry_type: SymmetryType = SymmetryType.NONE):
    """
    Save the solution to a file for later analysis.
    
    Args:
        n: Grid size
        point_vars: Dictionary of point variables from optimization
        objective_value: Optimal objective value
        filename: Output filename (optional)
        symmetry_type: Symmetry type used in solving
    """
    if filename is None:
        sym_suffix = f"_{symmetry_type.value}" if symmetry_type != SymmetryType.NONE else ""
        filename = f"no_three_in_line_n{n}{sym_suffix}_solution.txt"
    
    with open(filename, 'w') as f:
        f.write(f"No-three-in-line solution for {n}x{n} grid\n")
        f.write(f"Symmetry constraint: {symmetry_type.value}\n")
        f.write(f"Maximum number of points selected: {int(objective_value)}\n")
        f.write(f"Theoretical maximum of 2n = {2*n} points\n\n")
        
        # Write grid
        for i in range(n):
            row = []
            for j in range(n):
                if point_vars[(i, j)].x > 0.5:
                    row.append("O")
                else:
                    row.append(".")
            f.write(" ".join(row) + "\n")
        
        # Write selected points as coordinates
        f.write("\nSelected points (coordinates):\n")
        selected_points = []
        for i in range(n):
            for j in range(n):
                if point_vars[(i, j)].x > 0.5:
                    selected_points.append((i, j))
        
        for point in selected_points:
            f.write(f"{point}\n")
        
        # Write symmetry analysis
        f.write("\nSymmetry Analysis:\n")
        symmetries = check_solution_symmetry(n, point_vars)
        for sym_type, has_symmetry in symmetries.items():
            status = "Yes" if has_symmetry else "No"
            f.write(f"{sym_type.replace('_', ' ').title()}: {status}\n")
    
    print(f"Solution saved to {filename}")


@app.command()
def solve(
    n: Annotated[int, typer.Option(help="Size of the grid (n x n)")],
    symmetry: Annotated[SymmetryType, typer.Option(help="Type of symmetry to enforce")] = SymmetryType.NONE,
    threads: Annotated[int, typer.Option(help="Number of threads to use")] = 12,
    save_file: Annotated[str, typer.Option(help="Output file to save solution")] = None,
    time_limit: Annotated[int, typer.Option(help="Time limit in seconds")] = None
):
    """
    Solve the no-three-in-line problem for an n*n grid with optional symmetry constraints.

    This finds the maximum number of points that can be placed on an n*n grid
    such that no three points are collinear (lie on the same straight line).
    
    Symmetry options:
    - none: No symmetry constraints
    - horizontal: Mirror symmetry across vertical axis
    - vertical: Mirror symmetry across horizontal axis  
    - both_axes: Both horizontal and vertical symmetry
    - diagonal: Mirror symmetry across main diagonal
    - anti_diagonal: Mirror symmetry across anti-diagonal
    - both_diagonals: Both diagonal symmetries
    - rotational_90: 4-fold rotational symmetry
    - rotational_180: 2-fold rotational symmetry
    - all: All symmetries combined
    """
    print(f"Solving no-three-in-line problem for {n}*{n} grid...")
    print(f"Symmetry constraint: {symmetry.value}")
    print(f"Using {threads} threads (M2 Max optimized)")

    # Step 1: Generate all grid points
    points = generate_grid_points(n)
    print(f"Generated {len(points)} grid points")

    # Step 2: Find all groups of collinear points
    collinear_groups = find_collinear_points(points)
    lines_with_3plus = sum(1 for pts in collinear_groups.values() if len(pts) > 2)
    print(f"Found {lines_with_3plus} lines with 3 or more points")

    # Step 3: Build optimization model with symmetry
    model, point_vars = build_optimization_model(
        points, collinear_groups, min_points=2*n, symmetry_type=symmetry, threads=threads
    )
    
    # Set time limit if specified
    if time_limit:
        model.setParam("TimeLimit", time_limit)
        print(f"Time limit set to {time_limit} seconds")

    # Step 4: Solve with timing decorator
    solve_optimization_model(model)

    # Step 5: Display and save results
    if model.status == GRB.OPTIMAL:
        display_solution(n, point_vars, model.ObjVal)
        if save_file:
            save_solution_to_file(n, point_vars, model.ObjVal, save_file, symmetry)
    elif model.status == GRB.TIME_LIMIT:
        print(f"Time limit reached. Best solution found: {model.ObjVal}")
        if save_file:
            save_solution_to_file(n, point_vars, model.ObjVal, save_file, symmetry)
    else:
        print(f"Optimization failed with status: {model.status}")


@app.command()
def compare_symmetries(
    n: Annotated[int, typer.Option(help="Size of the grid (n x n)")],
    threads: Annotated[int, typer.Option(help="Number of threads to use")] = 12,
    time_limit: Annotated[int, typer.Option(help="Time limit per symmetry in seconds")] = 300
):
    """
    Compare solutions for different symmetry types to find the most aesthetically pleasing ones.
    """
    symmetry_types = [
        SymmetryType.NONE,
        SymmetryType.HORIZONTAL,
        SymmetryType.VERTICAL,
        SymmetryType.BOTH_AXES,
        SymmetryType.DIAGONAL,
        SymmetryType.ROTATIONAL_180,
        SymmetryType.ALL
    ]
    
    results = {}
    
    print(f"Comparing symmetry types for {n}x{n} grid...")
    print(f"Using {threads} threads, {time_limit}s per symmetry type\n")
    
    for symmetry in symmetry_types:
        print(f"Solving with {symmetry.value} symmetry...")
        
        # Generate points and collinear groups
        points = generate_grid_points(n)
        collinear_groups = find_collinear_points(points)
        
        # Build and solve model
        model, point_vars = build_optimization_model(
            points, collinear_groups, min_points=2*n, 
            symmetry_type=symmetry, threads=threads
        )
        
        if time_limit:
            model.setParam("TimeLimit", time_limit)
        
        solve_optimization_model(model)
        
        if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            results[symmetry.value] = {
                'objective': model.ObjVal,
                'status': 'Optimal' if model.status == GRB.OPTIMAL else 'Time Limit',
                'point_vars': point_vars
            }
            
            # Save solution
            filename = f"solution_n{n}_{symmetry.value}.txt"
            save_solution_to_file(n, point_vars, model.ObjVal, filename, symmetry)
        else:
            results[symmetry.value] = {
                'objective': 0,
                'status': 'Failed',
                'point_vars': None
            }
        
        print(f"Result: {results[symmetry.value]['objective']} points ({results[symmetry.value]['status']})\n")
    
    # Summary
    print("=" * 60)
    print("SYMMETRY COMPARISON SUMMARY")
    print("=" * 60)
    for symmetry_name, result in results.items():
        print(f"{symmetry_name:20} | {result['objective']:3.0f} points | {result['status']}")


if __name__ == "__main__":
    app()