# GROMACS Trajectory Checklist

3-step sanity check before MD trajectory analysis.
Built from real research experience — COMFHA Research Group / SIMATEC.

## Why this exists

After running MD simulations, most people don't know where to start with analysis.
This script checks the 3 things you must verify before touching your trajectory.

## Checks

| # | Check | Threshold |
|---|-------|-----------|
| 1 | Energy convergence | Drift < 1% (last 20%) |
| 2 | Density | 1000 ± 50 kg/m³ |
| 3 | RMSD plateau | Std dev < 0.5 Å (last 20%) |

## Requirements

pip install numpy matplotlib pandas

## How to generate input files

gmx energy -f md.edr -o energy.xvg    # select Potential
gmx energy -f md.edr -o density.xvg   # select Density
gmx rms    -f md.xtc -s md.tpr -o rmsd.xvg

## Usage

python gromacs_checklist.py \
    --energy energy.xvg \
    --rmsd rmsd.xvg \
    --density density.xvg

## For non-water solvents

python gromacs_checklist.py \
    --energy energy.xvg \
    --rmsd rmsd.xvg \
    --density density.xvg \
    --density-target 789 \
    --density-tol 30

## Author

Chalakon Pornjariyawatch — DPST Scholar, COMFHA Research Group
