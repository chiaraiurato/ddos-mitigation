# DDoS Mitigation System Simulation

This is the final project for the **Performance Modeling of Computer Systems** class, University of Rome Tor Vergata (Master's Degree in Computer Engineering), September 2025.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Execution](#execution)

---

## Overview

The project simulates a DDoS mitigation system composed of:

- **Mitigation Center** (finite FIFO queue),
- **Web Server** (Processor Sharing),
- **Spike Servers** dynamically allocated (Processor Sharing) via **autoscaling** when the workload exceeds defined thresholds.

Two **model variants** are available:

- **BASE**: only Mitigation → Web/Spike;
- **IMPROVED**: adds an **ML-based Analysis Center** to support filtering.

The simulator supports both **finite horizon analysis** (Replications) and **infinite horizon analysis** (Batch Means), as well as **verification** with exponential distributions and **validation** on load multipliers (×2, ×5, ×10, ×40).

## Repository Structure

```
src/
  main.py                      # Entry point 
  controller/
    simulation.py             # run_simulation / run_finite_horizon / run_infinite_horizon / run_verification
  engineering/
    costants.py               # global costants (arrival/service, threshold, RNG seed, batch means, ecc.)
    statistics.py             # utility for Batch Means, autocorrelazione, CI
  library/                    # RNG and distribution (rngs, rvgs, rvms)
  model/
    job.py                    # Job definition
    mitigation_center.py      #  # finite FIFO queue and service
    processor_sharing_server.py # PS server for Web/Spike
    mitigation_manager.py     # autoscaling + routing
  plot/                       # output CSV files + scripts/figures
  utils/
    resolve_markov_chain.py   # Markov chain verification
acs_input/                    # CSV for autocorrelation analysis
```

## Requirements

- **Python** ≥ 3.10
- Recommended Python packages:
  - `simpy`, `numpy`, `pandas`, `matplotlib`

## Installation

```bash
# Clone repo
git clone <repo-url>
cd <repo>

# Install dependencies
pip install -U pip
pip install simpy numpy pandas matplotlib
```

## Esxecution

Run the **interactive menu**:

```bash
python src/main.py
```

You will be asked to select two options:

1. **model selection**: `A` (BASE) or `B` (IMPROVED).
2. **execution model**:
   - 0: simulation without attack (single)
   - 1: Verification (exponential distributions)
   - 2: Standard (hyperexponential distributions)
   - 3: Validation (×1, ×2, ×5, ×10, ×40)
   - 4: Transient Analysis (finite horizon)
   - 5: Finite Horizon (replications)
   - 6: Infinite Horizon (Batch Means)
   - 7: Exit
