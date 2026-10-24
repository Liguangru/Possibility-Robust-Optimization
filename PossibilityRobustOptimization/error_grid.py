"""Error grids for proton therapy robustness analysis.

Implements the three error grids described in
de Jong et al. (2026) Phys. Med. Biol. 71 095016:

1. Optimization grid: 21 scenarios  (section 2.1)
2. Evaluation grid:   28 scenarios  (Korevaar et al. 2019, section 2.1)
3. SPE grid:          33 x 7 = 231 scenarios (section 2.2)

Each grid is an ``(N, 4)`` array of ``[dx, dy, dz, range_error_pct]``
with setup errors in mm and range errors in %.

Running ``python error_grid.py`` prints a summary of each grid to the
terminal and writes the full scenario lists to xlsx files in ./output/.
Requires: numpy, openpyxl.
"""

from itertools import product
from pathlib import Path

import numpy as np
from openpyxl import Workbook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_xlsx(grid: np.ndarray, path: Path) -> None:
    """Write grid to xlsx with header dx_mm,dy_mm,dz_mm,range_error_pct."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.append(["dx_mm", "dy_mm", "dz_mm", "range_error_pct"])
    for row in np.round(grid, 4):
        ws.append(row.tolist())
    wb.save(path)


# ---------------------------------------------------------------------------
# 1. Optimization grid: 21 scenarios
# ---------------------------------------------------------------------------
def grid_21(
    setup_error_mm: float = 3.0,
    range_error_pct: float = 3.0,
) -> np.ndarray:
    """7 setup points (6 face centers + origin) x 3 range errors (-r, 0, +r)."""
    setup = np.zeros((7, 3))
    for i in range(3):
        for j, sign in enumerate((-1, 1)):
            setup[1 + 2 * i + j, i] = sign * setup_error_mm
    return np.hstack([
        np.repeat(setup, 3, axis=0),
        np.tile([-range_error_pct, 0.0, range_error_pct], len(setup))[:, None],
    ])


# ---------------------------------------------------------------------------
# 2. Evaluation grid: 28 scenarios (Korevaar et al. 2019)
# ---------------------------------------------------------------------------
def grid_28(
    setup_error_mm: float = 3.0,
    range_error_pct: float = 3.0,
) -> np.ndarray:
    """14 setup points (6 face centers + 8 vertices on sphere of radius
    setup_error_mm) x 2 range errors (-r, +r)."""
    s = setup_error_mm / np.sqrt(3.0)
    setup = []
    for i in range(3):  # 6 face centers
        for sign in (-1, 1):
            v = np.zeros(3)
            v[i] = sign * setup_error_mm
            setup.append(v)
    setup += [np.array(p) * s for p in product((-1, 1), repeat=3)]  # 8 vertices
    setup = np.array(setup)
    return np.vstack([
        np.hstack([setup, np.full((len(setup), 1), r)])
        for r in (-range_error_pct, range_error_pct)
    ])


# ---------------------------------------------------------------------------
# 3. SPE grid (de Jong et al. 2026, section 2.2)
# ---------------------------------------------------------------------------
def spe_grid(sigma_sys_setup: float = 0.92,
             sigma_rand_setup: float = 1.00,
             grid_order_setup: int = 5,
             sigma_range: float = 1.5,
             grid_order_range: int = 7,
             sigma_multiplier: float = 3.0
             ) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian setup grid clipped to sphere of radius sigma_multiplier *
    sigma_setup, crossed with grid_order_range range nodes spanning
    +/- sigma_multiplier * sigma_range.  Returns (grid, range_nodes)."""
    sigma_setup = np.sqrt(sigma_rand_setup**2 + sigma_sys_setup**2)
    emax_setup = sigma_multiplier * sigma_setup
    emax_range = sigma_multiplier * sigma_range

    levels = np.linspace(-emax_setup, emax_setup, grid_order_setup)
    pts = np.array(list(product(levels, repeat=3)))
    setup = pts[np.linalg.norm(pts, axis=1) <= emax_setup + 1e-9]

    ranges = np.linspace(-emax_range, emax_range, grid_order_range)
    grid = np.hstack([
        np.repeat(setup, len(ranges), axis=0),
        np.tile(ranges, (len(setup), 1)).reshape(-1, 1),
    ])
    return grid, ranges


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(name: str,
                 grid: np.ndarray,
                 csv_path: Path,
                 range_nodes: np.ndarray | None = None) -> None:
    """Print scenario count, save full grid to CSV."""
    print(f"{name}: {len(grid)} scenarios")
    if range_nodes is not None:
        print(f"  Range nodes (%): {np.round(range_nodes, 2).tolist()}")
    save_xlsx(grid, csv_path)
    print(f"  Saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
def main() -> None:
    out = Path(__file__).parent / "output"

    print_report("Optimization grid: 21 scenarios",
                 grid_21(setup_error_mm=3.0, range_error_pct=3.0),
                 out / "grid_21.xlsx")

    print_report("Evaluation grid: 28 scenarios (Korevaar 2019)",
                 grid_28(setup_error_mm=3.0, range_error_pct=3.0),
                 out / "grid_28.xlsx")

    # Defaults reproduce the paper's settings (33 setup x 7 range = 231)
    g_spe, r_spe = spe_grid(
        sigma_sys_setup=0.92,
        sigma_rand_setup=1.00,
        grid_order_setup=5,
        sigma_range=1.5,
        grid_order_range=7,
        sigma_multiplier=3.0,
    )
    print_report("SPE grid (de Jong et al. 2026)",
                 g_spe, out / "grid_spe.xlsx", range_nodes=r_spe)


if __name__ == "__main__":
    main()
