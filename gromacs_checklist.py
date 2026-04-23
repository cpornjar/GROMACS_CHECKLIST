"""
GROMACS Trajectory Checklist
==============================
Checks 3 essential criteria before MD trajectory analysis:
  1. Energy convergence
  2. Density (~1000 kg/m³ for water systems)
  3. RMSD plateau

Author  : Chalakon Pornjariyawatch — COMFHA Research Group
GitHub  : https://github.com/cpornjar/gromacs-checklist
License : MIT

Requirements:
    pip install numpy matplotlib pandas

Usage:
    python gromacs_checklist.py \\
        --energy energy.xvg \\
        --rmsd rmsd.xvg \\
        --density density.xvg

How to generate input files from GROMACS:
    gmx energy  -f md.edr  -o energy.xvg     (select Potential)
    gmx rms     -f md.xtc  -s md.tpr -o rmsd.xvg
    gmx energy  -f md.edr  -o density.xvg    (select Density)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Thresholds ─────────────────────────────────────────────────────────────────
DENSITY_TARGET   = 1000.0   # kg/m³  (pure water, 298 K, 1 atm)
DENSITY_TOL      = 50.0     # ± kg/m³
RMSD_PLATEAU_TOL = 0.05     # nm  — std dev of last 20% must be below this
ENERGY_DRIFT_TOL = 0.01     # relative drift of last 20% vs full mean

# ── Terminal colours ────────────────────────────────────────────────────────────
PASS  = "\033[92m✓ PASS\033[0m"
FAIL  = "\033[91m✗ FAIL\033[0m"
WARN  = "\033[93m⚠ WARN\033[0m"
BOLD  = "\033[1m"
RESET = "\033[0m"


# ── XVG parser ─────────────────────────────────────────────────────────────────
def parse_xvg(filepath: str) -> pd.DataFrame:
    """Parse a GROMACS .xvg file, skipping comment/label lines."""
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(("#", "@")) or not line:
                continue
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise ValueError(f"No data found in {filepath}")
    df = pd.DataFrame(rows)
    df.columns = ["time"] + [f"col{i}" for i in range(1, df.shape[1])]
    return df


# ── Check 1 : Energy convergence ───────────────────────────────────────────────
def check_energy(filepath: str) -> dict:
    """
    Relative drift of the last 20% of the potential energy trace
    should be < ENERGY_DRIFT_TOL (1% by default).
    """
    df          = parse_xvg(filepath)
    vals        = df["col1"].values
    n           = len(vals)
    tail        = vals[int(0.80 * n):]
    global_mean = np.mean(vals)
    tail_mean   = np.mean(tail)
    tail_std    = np.std(tail)
    drift       = abs(tail_mean - global_mean) / abs(global_mean) if global_mean != 0 else 0

    return {
        "passed"      : drift < ENERGY_DRIFT_TOL,
        "drift_pct"   : drift * 100,
        "tail_mean"   : tail_mean,
        "tail_std"    : tail_std,
        "global_mean" : global_mean,
        "time"        : df["time"].values,
        "values"      : vals,
    }


# ── Check 2 : Density ──────────────────────────────────────────────────────────
def check_density(filepath: str) -> dict:
    """
    Mean density of the last 20% should be within DENSITY_TARGET ± DENSITY_TOL.
    """
    df   = parse_xvg(filepath)
    vals = df["col1"].values
    n    = len(vals)
    tail = vals[int(0.80 * n):]
    mean = np.mean(tail)

    return {
        "passed"    : abs(mean - DENSITY_TARGET) <= DENSITY_TOL,
        "mean"      : mean,
        "target"    : DENSITY_TARGET,
        "tolerance" : DENSITY_TOL,
        "time"      : df["time"].values,
        "values"    : vals,
    }


# ── Check 3 : RMSD plateau ─────────────────────────────────────────────────────
def check_rmsd(filepath: str) -> dict:
    """
    Std dev of the last 20% of RMSD should be < RMSD_PLATEAU_TOL.
    A plateau indicates the system is equilibrated.
    """
    df        = parse_xvg(filepath)
    vals      = df["col1"].values
    n         = len(vals)
    tail      = vals[int(0.80 * n):]
    tail_std  = np.std(tail)
    tail_mean = np.mean(tail)

    return {
        "passed"    : tail_std < RMSD_PLATEAU_TOL,
        "tail_std"  : tail_std,
        "tail_mean" : tail_mean,
        "threshold" : RMSD_PLATEAU_TOL,
        "time"      : df["time"].values,
        "values"    : vals,
    }


# ── Report ─────────────────────────────────────────────────────────────────────
def print_report(energy_r: dict, density_r: dict, rmsd_r: dict):
    width = 62
    print()
    print("=" * width)
    print(f"{BOLD}  GROMACS Trajectory Checklist  |  COMFHA{RESET}")
    print("=" * width)

    # [1] Energy
    s = PASS if energy_r["passed"] else FAIL
    print(f"\n{BOLD}[1] Energy Convergence{RESET}  {s}")
    print(f"    Global mean      : {energy_r['global_mean']:.2f} kJ/mol")
    print(f"    Tail mean        : {energy_r['tail_mean']:.2f} kJ/mol")
    print(f"    Drift (last 20%) : {energy_r['drift_pct']:.3f}%"
          f"  (threshold < {ENERGY_DRIFT_TOL*100:.1f}%)")
    if not energy_r["passed"]:
        print(f"    {WARN} Energy still drifting — consider extending simulation.")

    # [2] Density
    s    = PASS if density_r["passed"] else FAIL
    diff = density_r["mean"] - density_r["target"]
    print(f"\n{BOLD}[2] Density{RESET}  {s}")
    print(f"    Mean (last 20%)  : {density_r['mean']:.1f} kg/m³")
    print(f"    Target           : {density_r['target']:.1f} ± {density_r['tolerance']:.0f} kg/m³")
    print(f"    Deviation        : {diff:+.1f} kg/m³")
    if not density_r["passed"]:
        hint = ("too low — check water model or pressure coupling"
                if diff < 0 else "too high — check barostat settings")
        print(f"    {WARN} Density {hint}.")

    # [3] RMSD
    s = PASS if rmsd_r["passed"] else FAIL
    print(f"\n{BOLD}[3] RMSD Plateau{RESET}  {s}")
    print(f"    Mean RMSD (last 20%) : {rmsd_r['tail_mean']*10:.2f} Å")
    print(f"    Std dev              : {rmsd_r['tail_std']*10:.3f} Å"
          f"  (threshold < {rmsd_r['threshold']*10:.1f} Å)")
    if not rmsd_r["passed"]:
        print(f"    {WARN} RMSD not plateaued — system may still be equilibrating.")

    # Summary
    all_pass = all(r["passed"] for r in [energy_r, density_r, rmsd_r])
    n_fail   = sum(not r["passed"] for r in [energy_r, density_r, rmsd_r])
    print("\n" + "=" * width)
    if all_pass:
        print(f"  {BOLD}{PASS}  All 3 checks passed — safe to proceed with analysis.{RESET}")
    else:
        print(f"  {BOLD}{FAIL}  {n_fail}/3 check(s) failed — review before analysis.{RESET}")
    print("=" * width)
    print()


# ── Plot ───────────────────────────────────────────────────────────────────────
def save_plot(energy_r: dict, density_r: dict, rmsd_r: dict,
              outpath: str = "checklist_report.png"):
    fig = plt.figure(figsize=(12, 9), dpi=150)
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(3, 1, hspace=0.50)

    panels = [
        (energy_r,  "Potential Energy", "kJ/mol", "#5c9eff"),
        (density_r, "Density",          "kg/m³",  "#4ecb71"),
        (rmsd_r,    "RMSD",             "nm",     "#f5a623"),
    ]

    for i, (result, label, unit, color) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("#181c25")
        t      = result["time"]
        v      = result["values"]
        cutoff = int(0.80 * len(t))

        ax.plot(t[:cutoff], v[:cutoff], color=color, alpha=0.40,
                linewidth=0.8, label="0–80%")
        ax.plot(t[cutoff:], v[cutoff:], color=color, alpha=1.00,
                linewidth=1.2, label="80–100% (analysis window)")
        ax.axvline(t[cutoff], color="white", linewidth=0.6,
                   linestyle="--", alpha=0.35)

        if label == "Density":
            for y in [DENSITY_TARGET - DENSITY_TOL,
                      DENSITY_TARGET,
                      DENSITY_TARGET + DENSITY_TOL]:
                ax.axhline(y, color="#ff6b6b", linewidth=0.8,
                           linestyle=":", alpha=0.7)

        status_txt = "✓ PASS" if result["passed"] else "✗ FAIL"
        status_col = "#4ecb71" if result["passed"] else "#ff4f4f"
        ax.set_title(f"{label}  [{status_txt}]", color=status_col,
                     fontsize=10, fontweight="bold", loc="left", pad=5)
        ax.set_xlabel("Time (ps)", color="#999999", fontsize=8)
        ax.set_ylabel(unit,        color="#999999", fontsize=8)
        ax.tick_params(colors="#777777", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a3a")
        ax.legend(fontsize=7, loc="upper right",
                  facecolor="#1e1e2e", edgecolor="#333344",
                  labelcolor="#bbbbbb")

    fig.suptitle("GROMACS Trajectory Checklist  |  COMFHA",
                 color="#cccccc", fontsize=11, y=0.99)
    plt.savefig(outpath, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved → {outpath}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GROMACS 3-step trajectory checklist — COMFHA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with plot
  python gromacs_checklist.py --energy energy.xvg --rmsd rmsd.xvg --density density.xvg

  # Custom density target (e.g. non-aqueous solvent)
  python gromacs_checklist.py --energy energy.xvg --rmsd rmsd.xvg --density density.xvg \\
      --density-target 789 --density-tol 30

  # Skip plot
  python gromacs_checklist.py --energy energy.xvg --rmsd rmsd.xvg --density density.xvg --no-plot

Exit codes: 0 = all checks passed,  1 = one or more failed
        """,
    )
    p.add_argument("--energy",         required=True,
                   help=".xvg with Potential energy (from gmx energy)")
    p.add_argument("--rmsd",           required=True,
                   help=".xvg with RMSD (from gmx rms)")
    p.add_argument("--density",        required=True,
                   help=".xvg with Density (from gmx energy)")
    p.add_argument("--out",            default="checklist_report.png",
                   help="Output plot filename (default: checklist_report.png)")
    p.add_argument("--no-plot",        action="store_true",
                   help="Skip plot generation")
    p.add_argument("--density-target", type=float, default=DENSITY_TARGET,
                   help=f"Target density kg/m³ (default: {DENSITY_TARGET})")
    p.add_argument("--density-tol",    type=float, default=DENSITY_TOL,
                   help=f"Density tolerance kg/m³ (default: {DENSITY_TOL})")
    return p


def main():
    global DENSITY_TARGET, DENSITY_TOL

    parser = build_parser()
    args   = parser.parse_args()

    DENSITY_TARGET = args.density_target
    DENSITY_TOL    = args.density_tol

    for flag, path in [("--energy", args.energy),
                       ("--rmsd",   args.rmsd),
                       ("--density", args.density)]:
        if not Path(path).exists():
            print(f"Error: file not found for {flag}: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"\n  Reading energy   : {args.energy}")
    print(f"  Reading density  : {args.density}")
    print(f"  Reading RMSD     : {args.rmsd}")

    energy_r  = check_energy(args.energy)
    density_r = check_density(args.density)
    rmsd_r    = check_rmsd(args.rmsd)

    print_report(energy_r, density_r, rmsd_r)

    if not args.no_plot:
        save_plot(energy_r, density_r, rmsd_r, outpath=args.out)

    all_pass = energy_r["passed"] and density_r["passed"] and rmsd_r["passed"]
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
