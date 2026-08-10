# EnergyPlus assets

- `MEEB_building_model.idf` is the source building model.
- `Shawinigan_OpenMeteo_2025-12-01_to_2026-04-01.epw` supplies the weather file
  required to launch EnergyPlus.
- `Foundations_boundary_conditions.txt` supplies external foundation boundary schedules.
- `HeatingSP_custom_from_csv_2026_15min.txt` supplies the thermostat schedules
  referenced by the source IDF. The MPC creates a working IDF with controlled
  thermostat schedules replaced for Runtime API actuation.

Keep the two schedule files beside the IDF. The runner copies remaining
`Schedule:File` dependencies into its generated EnergyPlus work directory.
