# MEEB NODE/NCDE Economic MPC

This repository runs an economic model-predictive controller (MPC) against an
EnergyPlus model of the MEEB residence. A trained NODE/NCDE model predicts the
thermal and heating response of ten controlled zones. At each 15-minute
boundary, the controller optimizes a six-hour setpoint trajectory and applies
its first control action to EnergyPlus.
## Repository layout

```text
.
|-- MPC_Script.py                  Main controller and command-line entry point
|-- data/                          Simulation and demand-response input data
|-- energyplus/                    IDF, EPW, and Schedule:File dependencies
|-- models/                        Trained Equinox model
|-- outputs/                       Empty generated results folder
|-- scripts/check_repository.py    Fast input and schema validation
`-- requirements.txt               Python dependencies
```

All default paths are relative to the repository root. A clone can therefore be
moved without editing usernames or drive-specific paths.

## Requirements

- Python 3.11.14
- EnergyPlus 26.1.0 with its Python Runtime API (`pyenergyplus`)
- Internet access to the Open-Meteo Archive API during a simulation, a no internet backup is also implemented

The controller uses the archived Google Research Trajax project. The dependency
file installs it from its official GitHub repository.

## Installation

Create and activate a virtual environment, then install the dependencies:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:ENERGYPLUS_INSTALL_DIR = "C:\EnergyPlusV26-1-0"
```

### WSL/Linux

Install the Linux build of EnergyPlus inside WSL. Do not point WSL at the
Windows EnergyPlus DLLs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export ENERGYPLUS_INSTALL_DIR="$HOME/EnergyPlus-26-1-0"
```

For a CUDA-enabled JAX installation, follow the JAX installation instructions
for the CUDA version on the machine before installing the remaining packages.

## Validate the repository

This check does not launch EnergyPlus or require the scientific Python packages:

```text
python scripts/check_repository.py
```

It verifies required files, CSV schemas, and external files referenced by the
IDF.

## Run

Run every unique demand-response day listed in `data/evenements-pointe.csv`:

```text
python MPC_Script.py
```

Run one day in the current process:

```text
python MPC_Script.py --single-day 2025-12-03T00:00:00-05:00
```

Results are written beneath `outputs/dr_day_YYYYMMDD_mpc_light_bias_corrected/`.
The default batch mode launches each day in a fresh Python process to release
EnergyPlus and JAX native state between runs.

## Configuration

Core control settings such as the horizon, timestep, zone mappings, tariffs,
comfort bounds, and optimizer limits remain in the `USER SETTINGS` section of
`MPC_Script.py`.

Repository paths can be overridden without modifying the file:

| Environment variable | Purpose |
| --- | --- |
| `ENERGYPLUS_INSTALL_DIR` | EnergyPlus installation folder |
| `MEEB_IDF_PATH` | EnergyPlus IDF file |
| `MEEB_EPW_PATH` | EnergyPlus weather file |
| `MEEB_DR_EVENTS_PATH` | Demand-response event CSV |
| `MEEB_INITIAL_TEMPERATURES_PATH` | Initial zone-temperature CSV |
| `MEEB_TRAINING_DATA_PATH` | Simulation data used to reconstruct scalers |
| `MEEB_MODEL_PATH` | Serialized Equinox model |
| `MEEB_OUTPUT_DIR` | Generated output directory |
| `MEEB_LEGACY_BIAS_LOG_DIR` | Optional previous-run logs for bias correction |

## Citation

If you use any piece of the software of this repository please reference:
Gabriel Sabbagh, (2026), Modélisation thermique multizone de résidences par équations différentielles neuronales
[Ph.D. thesis].

