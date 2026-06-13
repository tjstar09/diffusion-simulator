#!/bin/bash
# ============================================================
# entrypoint.sh — Diffusion Simulation Docker Entrypoint
# ============================================================
# Usage: entrypoint.sh <command>
#   second_fixed     — Run the enhanced DDPM forward diffusion
#   run_simulations  — Run 100 simulations and generate CSV
#   analyze_simulations — Analyze CSV and produce analysis plot
#   all              — Run all three scripts in sequence
#   shell            — Start an interactive bash shell
#   (anything else)  — Pass through as a shell command
# ============================================================

set -e

cd /app

# If an output volume is mounted, restore any previously saved files
if [ -d /app/output ]; then
  cp -n /app/output/*.csv /app/ 2>/dev/null || true
  cp -n /app/output/*.png /app/ 2>/dev/null || true
fi

case "${1}" in
  second_fixed)
    echo "=== Running Enhanced Forward Diffusion (second_fixed.py) ==="
    python second_fixed.py
    if [ -d /app/output ]; then
      cp -n *.png /app/output/ 2>/dev/null || true
      echo "Output files copied to /app/output/"
    fi
    ;;
  run_simulations)
    echo "=== Running 100 Simulations (run_simulations.py) ==="
    python run_simulations.py
    if [ -d /app/output ]; then
      cp -n *.csv *.png /app/output/ 2>/dev/null || true
      echo "Output files copied to /app/output/"
    fi
    ;;
  analyze_simulations)
    echo "=== Running Analysis (analyze_simulations.py) ==="
    if [ ! -f diffusion_simulations.csv ]; then
      echo "ERROR: diffusion_simulations.csv not found."
      echo "Run 'run_simulations' first, or mount a volume with the CSV."
      exit 1
    fi
    python analyze_simulations.py
    if [ -d /app/output ]; then
      cp -n *.png /app/output/ 2>/dev/null || true
      echo "Output files copied to /app/output/"
    fi
    ;;
  all)
    echo "============================================================"
    echo "  Running ALL scripts in sequence"
    echo "============================================================"
    python second_fixed.py
    python run_simulations.py
    python analyze_simulations.py
    if [ -d /app/output ]; then
      cp -n * /app/output/ 2>/dev/null || true
      echo "All output files copied to /app/output/"
    fi
    echo "============================================================"
    echo "  All scripts completed."
    echo "============================================================"
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    # Pass-through: run whatever command was provided
    exec "$@"
    ;;
esac