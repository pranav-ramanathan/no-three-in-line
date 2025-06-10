from gurobipy import Model, GRB
from math import gcd
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Callable, Any
from typing_extensions import Annotated
from functools import wraps
import time

import typer

Point = Tuple[int, int]
Line = Tuple[int, int, int]  # Represents line equation ax + by + c = 0
PointSet = Set[Point]

app = typer.Typer()


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


def build_optimization_model(points: List[Point], 
                           collinear_groups: Dict[Line, PointSet],
                           min_points: int) -> Tuple[Model, Dict[Point, any]]:
    """
    Build the Integer Linear Programming model for the no-three-in-line problem.
    
    Variables: Binary variable for each grid point (1 = selected, 0 = not selected)
    Objective: Maximize number of selected points
    Constraints: 
        - At most 2 points selected on any line with 3+ points
        - At least min_points total points selected (optional tightening)
    
    Args:
        points: All grid points
        collinear_groups: Groups of collinear points
        min_points: Minimum number of points to select
        
    Returns:
        Tuple of (model, point_variables)
    """
    model = Model("No_Three_In_Line")
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output
    
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


def display_solution(n: int, point_vars: Dict[Point, any], objective_value: float):
    """
    Display the optimal solution as a grid.
    
    Args:
        n: Grid size
        point_vars: Dictionary of point variables from optimization
        objective_value: Optimal objective value
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


@app.command()
def solve(n: Annotated[int, typer.Option(help="Size of the grid (n x n)")]):
    """
    Solve the no-three-in-line problem for an n*n grid.
    
    This finds the maximum number of points that can be placed on an n*n grid
    such that no three points are collinear (lie on the same straight line).
    """
    print(f"Solving no-three-in-line problem for {n}*{n} grid...")
    
    # Step 1: Generate all grid points
    points = generate_grid_points(n)
    print(f"Generated {len(points)} grid points")
    
    # Step 2: Find all groups of collinear points  
    collinear_groups = find_collinear_points(points)
    lines_with_3plus = sum(1 for pts in collinear_groups.values() if len(pts) > 2)
    print(f"Found {lines_with_3plus} lines with 3 or more points")
    
    # Step 3: Build optimization model
    model, point_vars = build_optimization_model(
        points, collinear_groups, min_points=2*n
    )
    
    # Step 4: Solve with timing decorator
    solve_optimization_model(model)
    
    # Step 5: Display results
    if model.status == GRB.OPTIMAL:
        display_solution(n, point_vars, model.ObjVal)
    else:
        print(f"Optimization failed with status: {model.status}")


if __name__ == "__main__":
    typer.run(solve)
