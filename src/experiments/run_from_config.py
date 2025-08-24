"""
Configuration-driven experiment pipeline orchestrator.

Executes complete PINN vs FEM comparison experiments from YAML configuration files,
coordinating FEM solving, PINN training, dataset generation, black-box training,
and visualization generation in a reproducible and automated workflow.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def run(cmd_list, dry=False):
    """
    Execute a command with optional dry-run mode.

    Args:
        cmd_list: List of command arguments to execute
        dry: If True, only print the command without executing

    Returns:
        Return code from subprocess execution (0 for dry runs)
    """
    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd_list)
    print(f"[run] {cmd_str}")
    if dry:
        return 0
    return subprocess.run(cmd_list, check=True).returncode

def add_if(d: dict, key: str, flag: str, target: list):
    """
    Conditionally add command-line flag and value to target list.

    Args:
        d: Dictionary to check for key presence
        key: Key to look for in dictionary
        flag: Command-line flag to add (e.g., "--alpha0")
        target: Target list to append flag and value to
    """
    if key in d and d[key] is not None:
        target += [flag, str(d[key])]

def build_dataset_path(ds_cfg: dict) -> str:
    """
    Generate dataset file path based on configuration parameters.

    Constructs standardized dataset filename following the naming convention
    used by data_gen.py for consistent file organization.

    Args:
        ds_cfg: Dataset configuration dictionary with keys:
            P_low, P_high: Load parameter range
            n_configs: Number of parameter configurations
            M: Number of spatial points per configuration
            sigma: Noise level
            seed: Random seed (optional, defaults to 0)

    Returns:
        Standardized dataset file path string
    """
    # Matches the naming scheme used by your data_gen.py
    P_low, P_high = ds_cfg["P_low"], ds_cfg["P_high"]
    n_cfg, M = ds_cfg["n_configs"], ds_cfg["M"]
    sigma = ds_cfg["sigma"]
    seed = ds_cfg.get("seed", 0)
    return f"data/supervised/dataset_P_{P_low:.2f}_{P_high:.2f}_N{n_cfg}_M{M}_sigma{sigma:.3f}_seed{seed}.npz"

def main():
    """
    Main orchestration function for configuration-driven experiments.

    Parses YAML configuration files and coordinates execution of:
    1. FEM reference solution computation
    2. PINN training with physics-informed loss
    3. Supervised dataset generation for black-box training
    4. Black-box neural network training and evaluation
    5. Hero figure generation with comparative visualization

    Supports flexible stage selection, dry-run mode, and comprehensive
    boundary condition specifications including Dirichlet, Neumann, and Robin types.

    Configuration Structure:
        fem: FEM solver configuration (case, boundary conditions)
        pinn: PINN training configuration (epochs, loss weights, normalization)
        dataset: Supervised dataset generation parameters
        bb: Black-box network training configuration
        hero: Visualization generation settings
    """
    ap = argparse.ArgumentParser(description="Run an example pipeline from a YAML config.")
    ap.add_argument("--cfg", required=True, help="YAML file in examples/")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    ap.add_argument("--only", nargs="*", choices=["fem","pinn","dataset","bb","hero"],
                    help="Run only these stages (default: all).")
    ap.add_argument("--skip", nargs="*", choices=["fem","pinn","dataset","bb","hero"],
                    help="Skip these stages.")
    args = ap.parse_args()

    cfg_path = Path(args.cfg).resolve()
    if not cfg_path.exists():
        print(f"[error] config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text())
    outdir = Path(cfg.get("outdir", "data/outputs/example")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- FEM ----
    fem = cfg["fem"]
    # required: case, left_type, right_type
    fem_cmd = ["python", "-m", "src.training.run_fem",
               "--case", str(fem["case"]),
               "--left-type", str(fem["left_type"]),
               "--right-type", str(fem["right_type"]),
               "--out", str(outdir)]
    # optional BC params
    add_if(fem, "u0", "--u0", fem_cmd)          # Dirichlet left
    add_if(fem, "uL", "--uL", fem_cmd)          # Dirichlet right
    add_if(fem, "P0", "--P0", fem_cmd)          # Neumann left load
    add_if(fem, "PL", "--PL", fem_cmd)          # Neumann right load
    # Robin left boundary conditions
    add_if(fem, "alpha0", "--alpha0", fem_cmd)  # Robin left alpha coeff
    add_if(fem, "beta0", "--beta0", fem_cmd)    # Robin left beta coeff
    add_if(fem, "g0", "--g0", fem_cmd)          # Robin left inhomogeneity
    # Robin right boundary conditions
    add_if(fem, "alphaL", "--alphaL", fem_cmd)  # Robin right alpha coeff
    add_if(fem, "betaL", "--betaL", fem_cmd)    # Robin right beta coeff
    add_if(fem, "gL", "--gL", fem_cmd)          # Robin right inhomogeneity
    # Legacy parameter mapping for compatibility
    add_if(fem, "H_left", "--alpha0", fem_cmd)  # H_left -> alpha0
    add_if(fem, "G_left", "--g0", fem_cmd)      # G_left -> g0
    add_if(fem, "H_right", "--alphaL", fem_cmd) # H_right -> alphaL
    add_if(fem, "G_right", "--gL", fem_cmd)     # G_right -> gL

    # ---- PINN ----
    pinn = cfg["pinn"]
    pinn_cmd = ["python", "-m", "src.training.run_pinn",
                "--case", str(fem["case"]),
                "--epochs", str(pinn["epochs"]),
                "--w_bc", str(pinn["w_bc"]),
                "--out", str(outdir),
                "--left-type", str(fem["left_type"]),
                "--right-type", str(fem["right_type"])]
    add_if(fem, "u0", "--u0", pinn_cmd)
    add_if(fem, "uL", "--uL", pinn_cmd)
    add_if(fem, "P0", "--P0", pinn_cmd)
    add_if(fem, "PL", "--PL", pinn_cmd)
    # Robin boundary conditions - use FEM parameter names
    add_if(fem, "alpha0", "--alpha0", pinn_cmd)
    add_if(fem, "beta0", "--beta0", pinn_cmd)
    add_if(fem, "g0", "--g0", pinn_cmd)
    add_if(fem, "alphaL", "--alphaL", pinn_cmd)
    add_if(fem, "betaL", "--betaL", pinn_cmd)
    add_if(fem, "gL", "--gL", pinn_cmd)
    # Legacy parameter mapping for compatibility
    add_if(fem, "H_left", "--alpha0", pinn_cmd)
    add_if(fem, "G_left", "--g0", pinn_cmd)
    add_if(fem, "H_right", "--alphaL", pinn_cmd)
    add_if(fem, "G_right", "--gL", pinn_cmd)

    # optional normalizations - use correct flag format
    if pinn.get("normalize_x", True):
        pinn_cmd += ["--normalize_x"]
    else:
        pinn_cmd += ["--no-normalize_x"]

    if pinn.get("normalize_u", True):
        pinn_cmd += ["--normalize_u"]
    else:
        pinn_cmd += ["--no-normalize_u"]

    # Note: PINN script doesn't support --lr, --n-collocations, --snap-every, or --seed
    # These are hardcoded in the PINN implementation

    # ---- Dataset (for BB) ----
    ds = cfg.get("dataset", None)
    bb = cfg.get("bb", None)
    dataset_cmd = None
    ds_path = None
    if ds:
        ds_path = build_dataset_path(ds)
        dataset_cmd = ["python", "-m", "src.experiments.data_gen",
                       "--n-configs", str(ds["n_configs"]),
                       "--n-points", str(ds["M"]),
                       "--P-low", str(ds["P_low"]),
                       "--P-high", str(ds["P_high"]),
                       "--sigma", str(ds["sigma"]),
                       "--out", "data/supervised"]
        add_if(ds, "seed", "--seed", dataset_cmd)

    # ---- Black-box train ----
    bb_cmd = None
    if bb:
        if ds_path is None:
            print("[warn] bb present but dataset missing; skipping BB stage.")
        else:
            bb_cmd = ["python", "-m", "src.training.train_blackbox",
                      "--data", ds_path,
                      "--epochs", str(bb["epochs"]),
                      "--lr", str(bb.get("lr", 1e-3)),
                      "--width", str(bb.get("width", 64)),
                      "--depth", str(bb.get("depth", 4)),
                      "--p-eval", str(fem.get("PL", 0.0)),   # evaluate at same P as FEM/PINN right tip
                      "--snap-every", str(bb.get("snap", 50)),
                      "--model-out", str(outdir / "bb_model.pt"),
                      "--log-out",   str(outdir / "bb_trainlog.npz")]

    # ---- Hero figure ----
    hero = cfg.get("hero", {"enabled": True})
    hero_cmd = None
    if hero.get("enabled", True):
        hero_cmd = ["python", "-m", "src.visualization.hero_figure",
                    "--in-fem",  str(outdir / f"fem_{fem['case']}.csv"),
                    "--in-pinn", str(outdir / f"pinn_{fem['case']}_trainlog.npz"),
                    "--in-bb",   str(outdir / "bb_trainlog.npz"),
                    "--out",     str(outdir / "hero")]
        # optional metrics.csv path for reliability panel
        metrics_csv = hero.get("metrics_csv", "data/outputs/metrics.csv")
        if metrics_csv and Path(metrics_csv).exists():
            hero_cmd += ["--metrics-csv", metrics_csv]
        # optional overrides for reliability M/sigma
        if "rel_M" in hero: hero_cmd += ["--rel-M", str(hero["rel_M"])]
        if "rel_sigma" in hero: hero_cmd += ["--rel-sigma", str(hero["rel_sigma"])]

    # -------- Orchestration switches --------
    stages = ["fem","pinn","dataset","bb","hero"]
    def wanted(stage):
        if args.only and stage not in args.only: return False
        if args.skip and stage in args.skip: return False
        return True

    # -------- Execute --------
    if wanted("fem"):     run(fem_cmd, args.dry_run)
    if wanted("pinn"):    run(pinn_cmd, args.dry_run)
    if dataset_cmd and wanted("dataset"): run(dataset_cmd, args.dry_run)
    if bb_cmd and wanted("bb"):           run(bb_cmd, args.dry_run)
    if hero_cmd and wanted("hero"):       run(hero_cmd, args.dry_run)

if __name__ == "__main__":
    main()
