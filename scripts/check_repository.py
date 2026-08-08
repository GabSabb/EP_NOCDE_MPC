"""Validate repository assets without importing the MPC's heavy dependencies."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    ROOT / "MPC_Script.py",
    ROOT / "requirements.txt",
    ROOT / "models" / "NOCDE_MEEB_Sim.eqx",
    ROOT / "data" / "simulated_data.csv",
    ROOT / "data" / "evenements-pointe.csv",
    ROOT / "data" / "initial_zone_temperatures_0000_all_dr_days_wide.csv",
    ROOT / "energyplus" / "MEEB2013_0deg.idf",
    ROOT / "energyplus" / "Shawinigan_OpenMeteo_2025-12-01_to_2026-04-01.epw",
    ROOT / "energyplus" / "FoundationsBD2013.txt",
    ROOT / "energyplus" / "HeatingSP_custom_from_csv_2026_15min.txt",
)

EXPECTED_CSV_COLUMNS = {
    ROOT / "data" / "evenements-pointe.csv": {
        "dateDebut",
        "dateFin",
    },
    ROOT / "data" / "initial_zone_temperatures_0000_all_dr_days_wide.csv": {
        "dr_day",
        "initial_time_local",
        "T_init_garage_degC",
        "T_init_base_1_degC",
        "T_init_base_2_degC",
        "T_init_kitchen_degC",
        "T_init_dining_degC",
        "T_init_bath_degC",
        "T_init_bed_1_degC",
        "T_init_bed_2_degC",
        "T_init_bed_3_degC",
        "T_init_living_degC",
    },
    ROOT / "data" / "simulated_data.csv": {
        "time",
        "Ext_Temp",
        "GHI",
        "temperature_garage",
        "power_garage",
        "setpoint_garage",
        "temperature_salon",
        "power_salon",
        "setpoint_salon",
    },
}


def csv_columns(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return set(next(csv.reader(handle)))


def idf_schedule_files(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    without_comments = "\n".join(line.split("!", 1)[0] for line in text.splitlines())
    dependencies: set[str] = set()
    for object_text in without_comments.split(";"):
        fields = [field.strip() for field in object_text.split(",")]
        if len(fields) >= 4 and fields[0].lower() == "schedule:file":
            dependencies.add(fields[3])
    return dependencies


def main() -> int:
    errors: list[str] = []

    for path in REQUIRED_FILES:
        if not path.is_file():
            errors.append(f"Missing required file: {path.relative_to(ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"Required file is empty: {path.relative_to(ROOT)}")

    for path, required_columns in EXPECTED_CSV_COLUMNS.items():
        if not path.is_file():
            continue
        try:
            missing = required_columns - csv_columns(path)
        except (OSError, StopIteration) as exc:
            errors.append(f"Could not read {path.relative_to(ROOT)}: {exc}")
            continue
        if missing:
            errors.append(
                f"{path.relative_to(ROOT)} is missing columns: {sorted(missing)}"
            )

    idf_path = ROOT / "energyplus" / "MEEB2013_0deg.idf"
    if idf_path.is_file():
        for dependency in sorted(idf_schedule_files(idf_path)):
            dependency_path = idf_path.parent / Path(dependency).name
            if not dependency_path.is_file():
                errors.append(f"Missing IDF Schedule:File dependency: {dependency}")

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Repository validation passed: {len(REQUIRED_FILES)} required files found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
