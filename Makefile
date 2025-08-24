.PHONY: help setup example inrange extrap hero sweep format test clean

# --------- User knobs (override at CLI: make inrange CFG=examples/xyz.yaml) ----------
PY      ?= python
CFG     ?= examples/tip_load_inrange.yaml

# --------- Help ----------
help:
	@echo "Targets:"
	@echo "  make example CFG=examples/<file>.yaml   # run any YAML example"
	@echo "  make inrange                            # tip load, P=0.60 (uses examples/tip_load_inrange.yaml)"
	@echo "  make extrap                             # tip load, P=1.20 (uses examples/tip_load_extrap.yaml)"
	@echo "  make hero OUT=<run_dir> CASE=<case>     # build 2x2 hero figure for a run dir"
	@echo "  make sweep                              # run metrics sweep (writes data/outputs/metrics.csv)"
	@echo "  make setup | format | test | clean      # dev utilities"

# --------- Install project in editable mode ----------
setup:
	$(PY) -m pip install -e .

# --------- Run from YAML (recommended path) ----------
example:
	$(PY) -m src.experiments.run_from_config --cfg $(CFG)

inrange:
	$(MAKE) example CFG=examples/tip_load_inrange.yaml

extrap:
	$(MAKE) example CFG=examples/tip_load_extrap.yaml

# --------- Build hero figure for an existing run dir ----------
# Usage:
#   make hero OUT=data/outputs/examples/tip_inrange CASE=tip_load
hero:
	@if [ -z "$(OUT)" ] || [ -z "$(CASE)" ]; then \
		echo "Usage: make hero OUT=<run_dir> CASE=<case>"; exit 2; \
	fi
	$(PY) -m src.visualization.hero_figure \
	  --in-fem  $(OUT)/fem_$(CASE).csv \
	  --in-pinn $(OUT)/pinn_$(CASE)_trainlog.npz \
	  --in-bb   $(OUT)/bb_trainlog.npz \
	  --out     $(OUT)/hero

# --------- Experiment sweeps (matches README) ----------
sweep:
	$(PY) -m src.experiments.sweep_metrics \
	  --P 0.60 --M-list 5 10 20 40 80 --sigma-list 0.0 0.01 0.03 0.05 0.10 \
	  --seeds 0 1 2 --bb-epochs 3000 --pinn-epochs 3000 \
	  --out data/outputs

# --------- Dev utilities ----------
format:
	black src tests

test:
	$(PY) tests/test_physics.py
	$(PY) tests/test_shapes.py

clean:
	rm -rf data/outputs/*/*.png data/outputs/*/*.svg
