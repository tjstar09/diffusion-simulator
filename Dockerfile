# ============================================================
# Dockerfile — Diffusion Simulation Project
# ============================================================
# Build:    docker build -t diffusion-sim .
#
# Run individual scripts:
#   docker run --rm -v "%cd%:/app/output" diffusion-sim second_fixed
#   docker run --rm -v "%cd%:/app/output" diffusion-sim run_simulations
#   docker run --rm -v "%cd%:/app/output" diffusion-sim analyze_simulations
#
# Run all three in sequence:
#   docker run --rm -v "%cd%:/app/output" diffusion-sim all
#
# PowerShell users replace "%cd%" with "${PWD}"
# ============================================================

FROM python:3.12-slim

WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project source files
COPY second_fixed.py .
COPY run_simulations.py .
COPY analyze_simulations.py .

# Entrypoint: run any of the three scripts by name, or "all"
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]